#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt,sin,cos,tan,log,hypot,asin,acos,atan,log10,e
import operator

class Operators:

    def __init__(self):
	self.__add 		= (operator.add,2)
	self.__mul 		= (operator.mul,2)
	self.__sub 		= (operator.sub,2)
	self.__gt  		= (operator.gt,2)
	self.__ge  		= (operator.le,2)
	self.__eq  		= (operator.eq,2)
	self.__lt  		= (operator.lt,2)
	self.__le  		= (operator.le,2)
	self.__sin 		= (sin,1)
	self.__cos 		= (cos,1)
	self.__tan 		= (tan,1)
	self.__atan 		= (atan,1)
	self.__asin 		= (self.__asin__,1)
	self.__acos 		= (self.__acos__,1)
	self.__div 		= (self.__div__,2)
	self.__mod 		= (self.__mod__,2)
	self.__loge 		= (self.__loge__,1)
	self.__log10 		= (self.__log10__,1)
	self.__hypot 		= (hypot,2)
	self.__sigmoid 		= (self.__sigmoid__,1)
	self.__hyptan  		= (self.__hyptan__,1)
	self.__fast    		= (self.__fast__,1)
	self.__derivative_sigmoid = (self.__derivative_sigmoid__,1)
	self.__derivative_hyptan  = (self.__derivative_hyptan__,1)
	self.__operators          = [self.__add,self.__mul,self.__sub,
				     self.__gt,self.__ge,self.__eq,
				     self.__lt,self.__le,self.__sin,
				     self.__cos,self.__tan,self.__asin,
				     self.__acos,self.__atan,self.__div,
				     self.__mod,self.__loge,self.__log10,
				     self.__hypot,self.__sigmoid,self.__hyptan,
				     self.__fast,self.__derivative_sigmoid,
				     self.__derivative_hyptan]

    def __div__(self,a,b):
	try: return float(a)/b
	except ZeroDivisionError as zde: return 9999.0
    
    def __mod__(self,a,b):
	try: return float(a)%b
	except ZeroDivisionError as zde: return 9999.0
	
    def __asin__(self,a):
	try: return asin(a)
	except ValueError as ve: return 9999.0  
    
    def __acos__(self,a):
	try: return acos(a)
	except ValueError as ve: return 9999.0  
	
    def __loge__(self,a):
	try: return log(a)
	except ValueError as ve: return 9999.0
	
    def __log10__(self,a):
	try: return log10(a)
	except ValueError as ve: return 9999.0
	
    def __sigmoid__(self,a): 
	try: return 1.0/(1.0+(e**(-a)))
	except Exception as e: return 9999.0
    
    def __hyptan__(self,a):
	try: return ((e**a)-(e**(-a)))/((e**a)+(e**(-a)))
	except Exception as e: return 9999.0
	
    def __fast__(self,a):
	try: return a/(1.0+abs(a))
	except Exception as e: return 9999.0
	
    def __derivative_sigmoid__(self,a):
	try: return self.__sigmoid__(a)*(1.0-self.__sigmoid__(a))
	except Exception as e: return 9999.0
	
    def __derivative_hyptan__(self,a):
	try: return 1.0 - pow(__hyptan__(a),2)
	except Exception as e: return 9999.0
    
    def _get_operators(self): return self.__operators
