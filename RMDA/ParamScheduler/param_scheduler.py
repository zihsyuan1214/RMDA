import torch
import warnings
import weakref

from functools import wraps
from collections import Counter
from bisect import bisect_right

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)

class _ParamScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch, base learning rates and base momentum
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
                group.setdefault('initial_momentum', group['momentum'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
                elif 'initial_momentum' not in group:
                    raise KeyError("param 'initial_momentum' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
                    
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.base_momentums = [group['initial_momentum'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `param_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr
    
    def get_last_momentum(self):
        """ Return last computed momentum by current scheduler.
        """
        return self._last_momentum

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError
        
    def get_momentum(self):
        # Compute momentum using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, lr))
            else:
                print('Epoch {:5d}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, group, lr))

    def print_momentum(self, is_verbose, group, momentum, epoch=None):
        """Display the current momentum.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting momentum'
                      ' of group {} to {:.4e}.'.format(group, momentum))
            else:
                print('Epoch {:5d}: adjusting momentum'
                      ' of group {} to {:.4e}.'.format(epoch, group, momentum))


    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after parameters scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`param_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first param_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `param_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `param_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_param_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                self.o._get_momentum_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                self.o._get_momentum_called_within_step = False

        with _enable_get_param_call(self):
            if epoch is None:
                self.last_epoch += 1
                lr_values = self.get_lr()
                momentum_values = self.get_momentum()
                
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    lr_values = self._get_closed_form_lr()
                else:
                    lr_values = self.get_lr()
                if hasattr(self, "_get_closed_form_momentum"):
                    momentum_values = self._get_closed_form_momentum()
                else:
                    momentum_values = self.get_momentum()

        for i, data in enumerate(zip(self.optimizer.param_groups, lr_values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
        for i, data in enumerate(zip(self.optimizer.param_groups, momentum_values)):
            param_group, momentum = data
            
            if momentum <= 1e0:
                param_group['momentum'] = momentum
            else:
                momentum = param_group['momentum'] = 1e0
                
            self.print_momentum(self.verbose, i, momentum, epoch)

        self._last_momentum = [group['momentum'] for group in self.optimizer.param_groups]
        
        """
        Restart the algorithm when the learning rate and momentum change.
        That is, setting iteration ,accum and gradient buffer to zeros, and 
        updating the initial point to the current point
        """
        if self.restart():
            for param_group in self.optimizer.param_groups:
                param_group['iteration'] = 0
                param_group['accum'] = 0.0 
                with torch.no_grad():
                    for p in param_group['params']:
                        self.optimizer.state[p]['initial_point'].copy_(p.clone().detach())
                        self.optimizer.state[p]['gradient_buffer'].copy_(torch.zeros_like(p))        
        
        
class MultiStepParam(_ParamScheduler):
    """
    Decreases the learning rate and increases the momentum 
    of each parameter group by gamma once the number of epoch reaches one of the milestones.
    When learning rate and momentum change, algorithm must be restarted.
    Notice that such change can happen simultaneously with other changes to the learning rate 
    and momentum from outside this scheduler. When last_epoch=-1, 
    sets initial lr as lr and initial momentum as momentum.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): factor of learning rate and momentum change.
            Default: 1e-1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.01 for all groups
        >>> # lr = 0.01 and momentum = 0.01 if epoch < 30
        >>> # lr = 0.001 and momentum = 0.1 if 30 <= epoch < 80
        >>> # lr = 0.0001 and momentum = 1.0 if epoch >= 80
        >>> scheduler = MultiStepParam(optimizer, milestones=[30, 80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, gamma=1e-1, last_epoch=-1, verbose=False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepParam, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]
    
    def get_momentum(self):
        if not self._get_momentum_called_within_step:
            warnings.warn("To get the last momentum computed by the scheduler, "
                          "please use `get_last_momentum()`.", UserWarning)
                
        if self.last_epoch not in self.milestones:
            return [group['momentum'] for group in self.optimizer.param_groups]
        return [group['momentum'] * (1.0/self.gamma) ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        milestones = list(sorted(self.milestones.elements()))
        return [base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
                for base_lr in self.base_lrs]
    
    def _get_closed_form_momentum(self):
        milestones = list(sorted(self.milestones.elements()))
        return [base_momentum * (1.0/self.gamma) ** bisect_right(milestones, self.last_epoch)
                for base_momentum in self.base_momentums]
    
    def restart(self):
        milestones = list(sorted(self.milestones.elements()))
        last_epoch = self.last_epoch
        if last_epoch in milestones:
            return True
        return False
