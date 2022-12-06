from django.urls import path


from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path(r'^morphology/q/', views.index, name='morphology_q'),
    path(r'^text_list/$', views.TextsListView.as_view(), name='texts'),
    path(r'^texts/(?P<pk>[0-9]*)/$', views.TextView.as_view(), name="text"),
    path(r'^morphology/$', views.search, name="morphology"),
]
 # <str:text>/