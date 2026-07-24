from django.urls import path

from . import views

urlpatterns = [
    path("", views.repo_list, name="repo_list"),
    path("repos/<int:repo_id>/", views.repo_detail, name="repo_detail"),
    path("repos/<int:repo_id>/ask/", views.ask_question, name="ask_question"),

    path("api/repos/", views.api_repo_list, name="api_repo_list"),
    path("api/repos/<int:repo_id>/", views.api_repo_detail, name="api_repo_detail"),
    path("api/repos/<int:repo_id>/query/", views.api_repo_query, name="api_repo_query"),
    path("api/repos/<int:repo_id>/graph/", views.api_repo_graph, name="api_repo_graph"),
]
