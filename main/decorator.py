from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import redirect


def anonymous_required(function=None, redirect_url=None):
    """
    Decorator for views that checks that the user is NOT logged in.
    Redirects to the redirect_url if not.
    """
    if not redirect_url:
        redirect_url = 'home'  # Or any other URL name you want to redirect to

    actual_decorator = user_passes_test(
        lambda u: not u.is_authenticated,
        login_url=redirect_url,
        redirect_field_name=None
    )

    if function:
        return actual_decorator(function)
    return actual_decorator