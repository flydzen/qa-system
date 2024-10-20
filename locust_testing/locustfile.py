import random

from locust import HttpUser, TaskSet, task, between


class UserBehavior(TaskSet):
    @task(1)
    def ask_questions(self):
        questions = [
            {"question": f"Test question {i}", "topic": random.choice(["sports", "business"])}
            for i in range(5)
        ]
        payload = {"questions": questions}
        with self.client.post("/ask", json=payload, stream=True, catch_response=True) as response:
            if response.status_code == 200:
                for chunk in response.iter_content():
                    pass


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 2)
