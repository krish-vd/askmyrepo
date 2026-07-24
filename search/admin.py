from django.contrib import admin

from .models import CodeChunk, Query, Repo

admin.site.register(Repo)
admin.site.register(CodeChunk)
admin.site.register(Query)
