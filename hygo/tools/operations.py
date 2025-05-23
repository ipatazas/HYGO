operations = {'op':['+','-','*','/','sin','cos','log','exp','tanh'],
              'n_args':[2,2,2,2,1,1,1,1,1],
              'expression':[('arg0+arg1'),('arg0-arg1'),('arg0*arg1'),('mydiv(arg0,arg1)'),
                            ('mysin(arg0)'),('mycos(arg0)'),('mylog(arg0)'),('myexp(arg0)'),('mytanh(arg0)')],
              'simplification_cond':[['(1-mynanchecker(arg0)) and (1-mynanchecker(arg1)) and isnumeric(arg0) and isnumeric(arg1)',
                                      '(1-mynanchecker(arg0)) and (myfloat(arg1)==0)',
                                      '(1-mynanchecker(arg1)) and (myfloat(arg0)==0)'],
                                     ['(1-mynanchecker(arg0)) and (1-mynanchecker(arg1)) and isnumeric(arg0) and isnumeric(arg1)',
                                      '(1-mynanchecker(arg0)) and (myfloat(arg1)==0)',
                                      '(1-mynanchecker(arg1)) and (myfloat(arg0)==0) and isnumeric(arg1)',
                                      'arg0==arg1'],
                                     ['(1-mynanchecker(arg0)) and (1-mynanchecker(arg1)) and isnumeric(arg0) and isnumeric(arg1)',
                                      '((1-mynanchecker(arg0)) and (myfloat(arg0)==0)) or ((1-mynanchecker(arg1)) and (myfloat(arg1)==0))',
                                      '(1-mynanchecker(arg0)) and (myfloat(arg0)==1)',
                                      '(1-mynanchecker(arg1)) and (myfloat(arg1)==1)'],
                                     ['(1-mynanchecker(arg0)) and (1-mynanchecker(arg1)) and isnumeric(arg0) and isnumeric(arg1)',
                                      '((1-mynanchecker(arg0)) and (myfloat(arg0)==0)) and ((1-mynanchecker(arg1)) and (myfloat(arg1)!=0))',
                                      '(1-mynanchecker(arg1)) and (myfloat(arg1)==1)',
                                      'not isnumeric(arg0) and (myfloat(arg1)==0)'],
                                     ['(1-mynanchecker(arg0)) and isnumeric(arg0)'],
                                     ['(1-mynanchecker(arg0)) and isnumeric(arg0)'],
                                     ['(1-mynanchecker(arg0)) and isnumeric(arg0)'],
                                     ['(1-mynanchecker(arg0)) and isnumeric(arg0)'],
                                     ['(1-mynanchecker(arg0)) and isnumeric(arg0)']],
              'simplification_action':[['str(round(float(arg0)+float(arg1),precission))',
                                        'arg0','arg1'],
                                       ['str(round(float(arg0)-float(arg1),precission))',
                                        'arg0','str(float(arg1)*-1)','str(round(0,precission))'],
                                       ['str(round(float(arg0)*float(arg1),precission))',
                                        'str(round(0,precission))','arg1','arg0'],
                                       ['str(round(mydiv(float(arg0),float(arg1)),precission))',
                                        'str(round(0,precission))','arg0','str(1e36)'],
                                       ['str(round(float(np.sin(float(arg0))),precission))'],
                                       ['str(round(float(np.cos(float(arg0))),precission))'],
                                       ['str(round(float(mylog(float(arg0))),precission))'],
                                       ['str(round(float(myexp(float(arg0))),precission))'],
                                       ['str(round(float(mytanh(float(arg0))),precission))'],],
              'complexity':[1,1,2,2,5,5,10,10,5]}