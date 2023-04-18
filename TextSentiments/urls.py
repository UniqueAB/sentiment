
from django.contrib import admin
from django.urls import path,include
from rest_framework.authtoken import views
from django.conf import settings
from django.conf.urls.static import static

from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from drf_yasg.views import get_schema_view as swagger_get_schema_view


admin.site.site_header = 'Sentiment 360SCRM'                    # default: "Django Administration"
admin.site.index_title = 'Sentiment 360SCRM'                 # default: "Site administration"
admin.site.site_title = 'Sentiment 360SCRM' # default: "Django site admin"

schema_view=swagger_get_schema_view(
    openapi.Info(
        title="Sentiments API",
        default_version='1.0.0',
        description="API documentation of App",
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)

urlpatterns = [
    path('', schema_view.with_ui('swagger', cache_timeout=0), name="swagger-schema"),
    path('admin/', admin.site.urls),
    path('api/',include('sentiments.urls')),
   # path('gettoken/', views.obtain_auth_token)
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
