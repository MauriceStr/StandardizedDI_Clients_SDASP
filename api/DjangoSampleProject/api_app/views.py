from django.shortcuts import render
from rest_framework.viewsets import ModelViewSet
# Create your views here.
from api_app.models import RawTransaction
#Django Authentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK, HTTP_400_BAD_REQUEST
from api_app.serializers import RawTransactionSerializer
class RawTransactionViewSet(ModelViewSet):
    """
    API endpoint that allows transactions to be viewed or edited.
    """
    permission_classes = [IsAuthenticated]
    queryset = RawTransaction.objects.all()
    serializer_class = RawTransactionSerializer

    def update(self, request, *args, **kwargs):
        return Response(status=HTTP_200_OK)

    def create(self, request, *args, **kwargs):
        headers = request.headers
        data = dict(request.data)
        serializer = RawTransactionSerializer(data=data)
        try:
            if serializer.is_valid(raise_exception=True):
                print("Data is valid")
                response_data = serializer.validated_data
                response = Response('Transaction successfully uploaded', status=HTTP_200_OK)
            else:
                response = Response(serializer.errors, status=HTTP_400_BAD_REQUEST)
        except Exception as e:
            print('Validation errors: {}'.format(serializer.errors))
            response = Response(str(e), status=HTTP_400_BAD_REQUEST)
        finally:
            return response
