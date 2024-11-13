import ray
ray.init(local_mode=True)

@ray.remote
def test_task():
    return "Ray is working!"

result = ray.get(test_task.remote())
print(result)  # Should print "Ray is working!"