from locust import HttpUser, task

class WebsiteUser(HttpUser):
    @task
    def predict(self):
        self.client.post("http://127.0.0.1:8000/predict", data={
             
            "text": "I want to fight someone and hit them!"
            })