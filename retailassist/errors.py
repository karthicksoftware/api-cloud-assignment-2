class ServiceError(RuntimeError):
    def __init__(self, component: str, message: str, status: int = 503):
        self.component, self.message, self.status = component, message, status
        super().__init__(message)

