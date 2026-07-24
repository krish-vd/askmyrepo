from django.db import models
from pgvector.django import VectorField

EMBEDDING_DIM = 384


class Repo(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        CLONING = "cloning", "Cloning"
        CHUNKING = "chunking", "Chunking"
        EMBEDDING = "embedding", "Embedding"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    url = models.URLField(max_length=500)
    name = models.CharField(max_length=255, blank=True)
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    created_at = models.DateTimeField(auto_now_add=True)
    error_message = models.TextField(blank=True)

    def __str__(self):
        return self.name or self.url


class CodeChunk(models.Model):
    class SymbolType(models.TextChoices):
        FUNCTION = "function", "Function"
        CLASS = "class", "Class"
        METHOD = "method", "Method"
        MODULE = "module", "Module"

    repo = models.ForeignKey(Repo, on_delete=models.CASCADE, related_name="chunks")
    file_path = models.CharField(max_length=1000)
    symbol_name = models.CharField(max_length=500, null=True, blank=True)
    symbol_type = models.CharField(max_length=20, choices=SymbolType.choices)
    start_line = models.IntegerField()
    end_line = models.IntegerField()
    source_code = models.TextField()
    embedding = VectorField(dimensions=EMBEDDING_DIM, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Best-effort call graph: populated by a regex heuristic in ingestion.py,
    # not a real static analyzer. See build_call_graph() for the tradeoff.
    calls = models.ManyToManyField(
        "self", symmetrical=False, related_name="called_by", blank=True
    )

    class Meta:
        indexes = [
            models.Index(fields=["repo", "file_path"]),
        ]

    def __str__(self):
        return f"{self.file_path}:{self.start_line} ({self.symbol_name or self.symbol_type})"


class Query(models.Model):
    repo = models.ForeignKey(Repo, on_delete=models.CASCADE, related_name="queries")
    question = models.TextField()
    answer = models.TextField(blank=True)
    temperature = models.FloatField(default=0.2)
    top_p = models.FloatField(default=0.9)
    top_k = models.IntegerField(default=40)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.question[:60]
