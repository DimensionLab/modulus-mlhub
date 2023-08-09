# modulus-mlhub

Hack to make Modulus containers run with DockerSpawner (or projects like [ML Hub](https://github.com/ml-tooling/ml-hub)).

Motivation: Modulus' container has `CMD` set to `null` istead of empty list `[]``, therefore starting it through ML Hub [crashes DockerSpawner](https://github.com/jupyterhub/jupyterhub/issues/3805).
