"""
Derived work of Chris Redford released at https://github.com/drgrib/dotmap under the following terms:

The MIT License (MIT)

Copyright (c) 2015 Chris Redford
Copyright (c) 2019 Graphcore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from inspect import ismethod
from json import dumps
from pprint import pprint
from collections import OrderedDict, MutableMapping


def dict_map(d=None):
    if d is None:
        d = {}
    return ConfigMap(d, _dynamic=False, _evaluate=False, _evaluated=True)


class ConfigMethod(object):

    def __init__(self, obj, method, args, definition):
        self._ = {'obj': obj, 'method': method, 'args': args, 'definition': definition}

    def evaluate(self):
        # disable config evaluation
        state = self._['obj'].config._evaluate
        self._['obj'].config._evaluate = False
        try:
            costate = self._['obj'].co.config._evaluate
            self._['obj'].co.config._evaluated = False
        except AttributeError:
            costate = None

        if self._['args'] == '':
            value = getattr(self._['obj'], self._['method'])()
        else:
            # todo: get rid of nasty eval by implementing a proper parser
            obj = self._['obj']
            value = eval('obj.' + self._['method'] + '(' + self._['args'] + ')')

        # reset config evaluation
        self._['obj'].config._evaluate = state
        if costate is not None:
            self._['obj'].co.config._evaluate = costate

        if isinstance(value, dict):
            return ConfigMap(value, _dynamic=False)

        return value

    def __repr__(self):
        return f"ConfigMethod(method={self._['obj'].__class__.__name__}.{self._['method']}, args='{self._['args']}')"

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args, **kwargs):
        return self.evaluate()


class ConfigMap(MutableMapping, OrderedDict):

    def __init__(self, *args, **kwargs):
        self._map = OrderedDict()
        # todo: simplify
        self._dynamic = True
        if kwargs:
            if '_dynamic' in kwargs:
                self._dynamic = kwargs['_dynamic']
                del kwargs['_dynamic']
        self._evaluate = True
        if kwargs:
            if '_evaluate' in kwargs:
                self._evaluate = kwargs['_evaluate']
                del kwargs['_evaluate']
        self._evaluated = False
        if kwargs:
            if '_evaluated' in kwargs:
                self._evaluated = kwargs['_evaluated']
                del kwargs['_evaluated']
        if args:
            d = args[0]
            if isinstance(d, dict):
                for k, v in self.__call_items(d):
                    if isinstance(v, dict):
                        v = ConfigMap(v, _dynamic=self._dynamic, _evaluate=self._evaluate, _evaluated=self._evaluated)
                    if type(v) is list:
                        l = []
                        for i in v:
                            n = i
                            if type(i) is dict:
                                n = ConfigMap(i, _dynamic=self._dynamic, _evaluate=self._evaluate,
                                              _evaluated=self._evaluated)
                            l.append(n)
                        v = l
                    self._map[k] = v
        if kwargs:
            for k, v in self.__call_items(kwargs):
                self._map[k] = v

    def __call_items(self, obj):
        if hasattr(obj, 'iteritems') and ismethod(getattr(obj, 'iteritems')):
            return obj.iteritems()
        else:
            return obj.items()

    def items(self):
        return self.iteritems()

    def iteritems(self):
        return self.__call_items(self._map)

    def __iter__(self):
        return self._map.__iter__()

    def next(self):
        return self._map.next()

    def __setitem__(self, k, v):
        self._map[k] = v

    def __getitem__(self, k, evaluate=None):
        if evaluate is None:
            evaluate = self._evaluate

        if k not in self._map:
            if k == '_ipython_canary_method_should_not_exist_':
                raise KeyError

            if self._dynamic:
                # automatically extend to new ConfigMap
                self[k] = ConfigMap()
            else:
                # todo: display full recursive path?
                raise KeyError("'%s' does not exist" % k)

        var = self._map[k]

        if evaluate:
            if isinstance(var, ConfigMethod):
                var = var.evaluate()
                # todo: return instead to avoid second config map eval?

            if isinstance(var, ConfigMap):
                var = var.evaluate()

        return var

    def __setattr__(self, k, v):
        if k in ['_map', '_dynamic', '_ipython_canary_method_should_not_exist_', '_evaluate', '_evaluated']:
            super(ConfigMap, self).__setattr__(k, v)
        else:
            self[k] = v

    def __getattr__(self, k):
        if k in ['_map', '_dynamic', '_ipython_canary_method_should_not_exist_', '_evaluate', '_evaluated']:
            return self.__getattribute__(k)
        else:
            return self[k]

    def __delattr__(self, key):
        return self._map.__delitem__(key)

    def __contains__(self, k):
        return self._map.__contains__(k)

    def __str__(self):
        items = []
        for k, v in self.__call_items(self._map):
            # bizarre recursive assignment situation (why someone would do this is beyond me)
            if id(v) == id(self):
                items.append('{0}=ConfigMap(...)'.format(k))
            else:
                items.append('{0}={1}'.format(k, repr(v)))
        joined = ', '.join(items)
        out = '{0}({1})'.format(self.__class__.__name__, joined)
        return out

    def __repr__(self):
        return str(self)

    def toDict(self, evaluate=None, with_hidden=True):
        if evaluate is None:
            evaluate = bool(self._evaluate)

        d = {}
        for k, v in self.items():
            if evaluate and isinstance(v, ConfigMethod):
                v = v.evaluate()
            if isinstance(v, ConfigMap):
                v = v.toDict(evaluate=evaluate, with_hidden=with_hidden) if id(v) != id(self) else d
            elif isinstance(v, list):
                v = [i.toDict(evaluate=evaluate, with_hidden=with_hidden) if isinstance(i, ConfigMap) else i for i in v]
            elif isinstance(v, tuple):
                v = (i.toDict(evaluate=evaluate, with_hidden=with_hidden) if isinstance(i, ConfigMap) else i for i in v)

            if with_hidden is False \
                    and (isinstance(k, str) and
                         ((k.startswith('_') and not k.endswith('_')) or k.startswith('~'))):
                continue

            d[k] = v

        return d

    def evaluate(self):
        if self._evaluated:
            return self

        # TODO: case where config method access a key of the config that is just being evaluated.
        #  shouldn't give an endless loop

        # todo: make more efficient
        return ConfigMap(self.toDict(evaluate=True), _dynamic=False, _evaluated=True)

    def pprint(self, pformat='json'):
        if pformat == 'json':
            print(dumps(self.toDict(), indent=4, sort_keys=True, default=str))
        else:
            pprint(self.toDict())

    def empty(self):
        return not any(self)

        # proper dict subclassing

    def values(self):
        return self._map.values()

        # ipython support

    def __dir__(self):
        return self.keys()

    @classmethod
    def parseOther(self, other):
        if type(other) is ConfigMap:
            return other._map
        else:
            return other

    def __cmp__(self, other):
        other = ConfigMap.parseOther(other)
        return self._map.__cmp__(other)

    def __eq__(self, other):
        other = ConfigMap.parseOther(other)
        if not isinstance(other, dict):
            return False
        return self._map.__eq__(other)

    def __ge__(self, other):
        other = ConfigMap.parseOther(other)
        return self._map.__ge__(other)

    def __gt__(self, other):
        other = ConfigMap.parseOther(other)
        return self._map.__gt__(other)

    def __le__(self, other):
        other = ConfigMap.parseOther(other)
        return self._map.__le__(other)

    def __lt__(self, other):
        other = ConfigMap.parseOther(other)
        return self._map.__lt__(other)

    def __ne__(self, other):
        other = ConfigMap.parseOther(other)
        return self._map.__ne__(other)

    def __delitem__(self, key):
        return self._map.__delitem__(key)

    def __len__(self):
        return self._map.__len__()

    def clear(self):
        self._map.clear()

    def copy(self):
        return ConfigMap(self, _dynamic=self._dynamic, _evaluate=self._evaluate, _evaluated=self._evaluated)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo=None):
        return self.copy()

    def get(self, key, default=None):
        return self._map.get(key, default)

    def has_key(self, key):
        return key in self._map

    def iterkeys(self):
        return self._map.iterkeys()

    def itervalues(self):
        return self._map.itervalues()

    def keys(self):
        return self._map.keys()

    def pop(self, key, default=None):
        return self._map.pop(key, default)

    def popitem(self):
        return self._map.popitem()

    def setdefault(self, key, default=None):
        self._map.setdefault(key, default)

    def update(self, *args, **kwargs):
        if len(args) != 0:
            self._map.update(*args)
        self._map.update(kwargs)

    def viewitems(self):
        return self._map.viewitems()

    def viewkeys(self):
        return self._map.viewkeys()

    def viewvalues(self):
        return self._map.viewvalues()

    @classmethod
    def fromkeys(cls, seq, value=None):
        d = ConfigMap(_dynamic=False)
        d._map = OrderedDict.fromkeys(seq, value)
        return d

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    # bannerStr
    def _getListStr(self, items):
        out = '['
        mid = ''
        for i in items:
            mid += '  {}\n'.format(i)
        if mid != '':
            mid = '\n' + mid
        out += mid
        out += ']'
        return out

    def _getValueStr(self, k, v):
        outV = v
        multiLine = len(str(v).split('\n')) > 1
        if multiLine:
            # push to next line
            outV = '\n' + v
        if type(v) is list:
            outV = self._getListStr(v)
        out = '{} {}'.format(k, outV)
        return out

    def _getSubMapDotList(self, pre, name, subMap):
        outList = []
        if pre == '':
            pre = name
        else:
            pre = '{}.{}'.format(pre, name)

        def stamp(pre, k, v):
            valStr = self._getValueStr(k, v)
            return '{}.{}'.format(pre, valStr)

        for k, v in subMap.items():
            if isinstance(v, ConfigMap) and v != ConfigMap():
                subList = self._getSubMapDotList(pre, k, v)
                outList.extend(subList)
            else:
                outList.append(stamp(pre, k, v))
        return outList

    def _getSubMapStr(self, name, subMap):
        outList = ['== {} =='.format(name)]
        for k, v in subMap.items():
            if isinstance(v, ConfigMap) and v != ConfigMap():
                # break down to dots
                subList = self._getSubMapDotList('', k, v)
                # add the divit
                # subList = ['> {}'.format(i) for i in subList]
                outList.extend(subList)
            else:
                out = self._getValueStr(k, v)
                # out = '> {}'.format(out)
                out = '{}'.format(out)
                outList.append(out)
        finalOut = '\n'.join(outList)
        return finalOut

    def bannerStr(self):
        lines = []
        previous = None
        for k, v in self.items():
            if previous == 'ConfigMap':
                lines.append('-')
            out = ''
            if isinstance(v, ConfigMap):
                name = k
                subMap = v
                out = self._getSubMapStr(name, subMap)
                lines.append(out)
                previous = 'ConfigMap'
            else:
                out = self._getValueStr(k, v)
                lines.append(out)
                previous = 'other'
        lines.append('--')
        s = '\n'.join(lines)

        return s
