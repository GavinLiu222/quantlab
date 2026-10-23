{% set public = namespace(methods=[], attributes=[], inherited=False) %}
{% for item in methods if not item.startswith('_') and item not in inherited_members %}
{% set public.methods = public.methods + [item] %}
{% endfor %}
{% for item in attributes if not item.startswith('_') and item not in inherited_members %}
{% set public.attributes = public.attributes + [item] %}
{% endfor %}
{% for item in methods + attributes if not item.startswith('_') and item in inherited_members %}
{% set public.inherited = True %}
{% endfor %}
{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :show-inheritance:

{% if public.inherited %}
Inherited methods and attributes are documented on the base classes linked above.
{% endif %}

{% if public.methods %}
Methods
-------

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in public.methods %}
   {{ objname }}.{{ item }}
{%- endfor %}
{% endif %}

{% if public.attributes %}
Attributes
----------
{% for item in public.attributes %}

.. autoattribute:: {{ objname }}.{{ item }}
{%- endfor %}
{% endif %}

{% if common_inherited_api is defined and common_inherited_api.get(fullname) %}
Common inherited methods
------------------------

These methods use the shared interface documented on the base class.

.. autosummary::
   :nosignatures:
{% for item in common_inherited_api[fullname] %}
   ~{{ item }}
{%- endfor %}
{% endif %}
