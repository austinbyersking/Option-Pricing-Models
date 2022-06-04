#Black Scholes Model 

#There are a number of assumptions to consider when viewing the formula
"""
1) Interest rate is known and constant through time. 

2) The stock follows a random walk in continuous time, the variance of the stock price paths follow a log-normal distribution. 

3) Volatility is constant 

4) Stock pays no dividends (can be modified to include them however)

5) The option can only be exercised at expiration i.e. it is a European type option.

6) No transaction costs i.e. fees on shorting selling etc. 

7) Fractional trading is possible i.e. we can buy/sell 0.x of any given stock. 
"""

#Important variables
"""
S : current asset price

K: strike price of the option

r: risk free rate 

T : time until option expiration 

σ: annualized volatility of the asset's returns 
"""


from scipy.stats import norm
import numpy as np

class BsOption:
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r 
        self.sigma = sigma
        self.q = q
        
    
    @staticmethod
    def N(x):
        return norm.cdf(x)
    
    @property
    def params(self):
        return {'S': self.S, 
                'K': self.K, 
                'T': self.T, 
                'r':self.r,
                'q':self.q,
                'sigma':self.sigma}
    
    def d1(self):
        return (np.log(self.S/self.K) + (self.r -self.q + self.sigma**2/2)*self.T) \
                                / (self.sigma*np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.T)
    
    def _call_value(self):
        return self.S*np.exp(-self.q*self.T)*self.N(self.d1()) - \
                    self.K*np.exp(-self.r*self.T) * self.N(self.d2())
                    
    def _put_value(self):
        return self.K*np.exp(-self.r*self.T) * self.N(-self.d2()) -\
                self.S*np.exp(-self.q*self.T)*self.N(-self.d1())
    
    def price(self, type_ = 'C'):
        if type_ == 'C':
            return self._call_value()
        if type_ == 'P':
            return self._put_value() 
        if type_ == 'B':
            return  {'call': self._call_value(), 'put': self._put_value()}
        else:
            raise ValueError('Unrecognized type')
            
        
        
        
        
if __name__ == '__main__':
    K = 100
    r = 0.05
    T = 1
    sigma = 0.25
    S = 50
    print(BsOption(S, K, T, r, sigma).price('B'))