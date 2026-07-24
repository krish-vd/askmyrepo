import json

from django.http import JsonResponse, HttpResponseNotAllowed
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt

from .ingestion import ingest_repo
from .models import CodeChunk, Query, Repo
from .retrieval import answer_question


# ---------------------------------------------------------------------------
# Browser-facing views (bare-bones templates, functional only)
# ---------------------------------------------------------------------------

def repo_list(request):
    if request.method == "POST":
        url = request.POST.get("url", "").strip()
        if url:
            repo = Repo.objects.create(url=url, status=Repo.Status.PENDING)
            # Synchronous ingest — see note in ingestion.ingest_repo() about
            # why this isn't a background task for this build.
            try:
                ingest_repo(repo)
            except Exception:
                pass  # status/error_message already recorded on repo by ingest_repo
            return redirect("repo_detail", repo_id=repo.id)
        return redirect("repo_list")

    repos = Repo.objects.order_by("-created_at")
    return render(request, "search/repo_list.html", {"repos": repos})


def repo_detail(request, repo_id):
    repo = get_object_or_404(Repo, id=repo_id)
    chunk_count = repo.chunks.count()
    file_count = repo.chunks.values("file_path").distinct().count()
    queries = repo.queries.order_by("-created_at")[:20]
    return render(request, "search/repo_detail.html", {
        "repo": repo,
        "chunk_count": chunk_count,
        "file_count": file_count,
        "queries": queries,
    })


def api_repo_graph(request, repo_id):
    """Lightweight node/edge list for the retrieval-graph visualization.
    Capped since large repos would make an unreadable/slow force layout."""
    repo = get_object_or_404(Repo, id=repo_id)
    chunks = list(repo.chunks.all()[:300])
    id_set = {c.id for c in chunks}
    nodes = [
        {"id": c.id, "label": c.symbol_name or c.symbol_type, "file": c.file_path}
        for c in chunks
    ]
    edges = []
    for c in chunks:
        for called in c.calls.filter(id__in=id_set):
            edges.append({"source": c.id, "target": called.id})
    return JsonResponse({"nodes": nodes, "edges": edges})


def ask_question(request, repo_id):
    repo = get_object_or_404(Repo, id=repo_id)
    if repo.status != Repo.Status.READY:
        return render(request, "search/ask.html", {
            "repo": repo,
            "error": f"Repo is not ready yet (status: {repo.status}).",
        })

    if request.method == "POST":
        question = request.POST.get("question", "").strip()
        temperature = float(request.POST.get("temperature", 0.2))
        top_p = float(request.POST.get("top_p", 0.9))
        top_k_sampling = int(request.POST.get("top_k_sampling", 40))
        top_k_retrieval = int(request.POST.get("top_k_retrieval", 4))

        if not question:
            return redirect("ask_question", repo_id=repo.id)

        result = answer_question(
            repo,
            question,
            top_k_retrieval=top_k_retrieval,
            temperature=temperature,
            top_p=top_p,
            top_k_sampling=top_k_sampling,
        )

        Query.objects.create(
            repo=repo,
            question=question,
            answer=result["answer"],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k_sampling,
        )

        return render(request, "search/results.html", {
            "repo": repo,
            "question": question,
            "result": result,
        })

    return render(request, "search/ask.html", {"repo": repo})


# ---------------------------------------------------------------------------
# JSON API (for the JS frontend that will be wired to the approved design)
# csrf_exempt: no auth/sessions gate these endpoints (out of scope per spec),
# so there's no session to forge a CSRF attack against; a production version
# with real auth would need proper CSRF/token handling instead.
# ---------------------------------------------------------------------------

@csrf_exempt
def api_repo_list(request):
    if request.method == "GET":
        repos = Repo.objects.order_by("-created_at").values(
            "id", "url", "name", "status", "created_at", "error_message"
        )
        return JsonResponse({"repos": list(repos)}, safe=False)

    if request.method == "POST":
        try:
            body = json.loads(request.body or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"error": "invalid JSON body"}, status=400)
        url = (body.get("url") or "").strip()
        if not url:
            return JsonResponse({"error": "url is required"}, status=400)

        repo = Repo.objects.create(url=url, status=Repo.Status.PENDING)
        try:
            ingest_repo(repo)
        except Exception as exc:
            return JsonResponse({
                "id": repo.id,
                "status": repo.status,
                "error_message": repo.error_message or str(exc),
            }, status=500)

        return JsonResponse({
            "id": repo.id,
            "name": repo.name,
            "status": repo.status,
            "chunk_count": repo.chunks.count(),
        }, status=201)

    return HttpResponseNotAllowed(["GET", "POST"])


def api_repo_detail(request, repo_id):
    repo = get_object_or_404(Repo, id=repo_id)
    return JsonResponse({
        "id": repo.id,
        "url": repo.url,
        "name": repo.name,
        "status": repo.status,
        "error_message": repo.error_message,
        "chunk_count": repo.chunks.count(),
        "file_count": repo.chunks.values("file_path").distinct().count(),
        "created_at": repo.created_at.isoformat(),
    })


@csrf_exempt
def api_repo_query(request, repo_id):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    repo = get_object_or_404(Repo, id=repo_id)
    if repo.status != Repo.Status.READY:
        return JsonResponse({"error": f"repo not ready (status: {repo.status})"}, status=409)

    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid JSON body"}, status=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JsonResponse({"error": "question is required"}, status=400)

    top_k_retrieval = int(body.get("top_k_retrieval", 4))
    temperature = float(body.get("temperature", 0.2))
    top_p = float(body.get("top_p", 0.9))
    top_k_sampling = int(body.get("top_k_sampling", 40))

    result = answer_question(
        repo,
        question,
        top_k_retrieval=top_k_retrieval,
        temperature=temperature,
        top_p=top_p,
        top_k_sampling=top_k_sampling,
    )

    Query.objects.create(
        repo=repo,
        question=question,
        answer=result["answer"],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k_sampling,
    )

    return JsonResponse(result)
