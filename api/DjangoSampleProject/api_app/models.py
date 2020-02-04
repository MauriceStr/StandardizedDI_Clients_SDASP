from django.db import models
from datetime import date

# Create your models here.
class RawTransaction(models.Model):
    transaction_id = models.CharField(max_length=100, null=True, blank=True)
    worth = models.CharField(max_length=300, null=True, blank=True)
    currency = models.CharField(max_length=50, null=True, blank=True)
    amount = models.CharField(max_length=300, null=True, blank=True)
    unit = models.CharField(max_length=80, null=True, blank=True)
    date = models.DateField(default= date.today, null=True, blank=True)
    prod_id = models.CharField(max_length=100, null=True, blank=True)
    cust_id = models.CharField(max_length=100, null=True, blank=True)
    load_timestamp = models.DateTimeField(auto_now=True)