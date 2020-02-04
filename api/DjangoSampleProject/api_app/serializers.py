from rest_framework import serializers
from api_app.models import RawTransaction


class RawTransactionSerializer(serializers.ModelSerializer):
    transaction_id = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    net_worth = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    worth = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    currency = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    amount = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    unit = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    cust_id = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    prod_id = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    date = serializers.CharField(required=False, allow_null=True, allow_blank=True)

    class Meta:
        model = RawTransaction
        fields = (
            'transaction_id',
            'worth',
            'currency',
            'amount',
            'unit',
            'cust_id',
            'prod_id',
            'date'
        )

    def validate(self, data):
        worth = data.get('worth', None)

        if worth < 0:
            raise serializers.ValidationError('No negative transactions allowed')

        if worth == 0:
            raise serializers.ValidationError('No zero value transactions allowed')

        # in case worth is a positve value, we assume the data to be vbalid
        return data

    # method automatically called in case of a post request
    def create(self, validated_data):
        params = validated_data
        return RawTransaction.objects.create(**params)

