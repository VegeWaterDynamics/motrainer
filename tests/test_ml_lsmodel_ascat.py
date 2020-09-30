#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the ml_lsmodel_ascat module.
"""
import pytest

from ml_lsmodel_ascat import ml_lsmodel_ascat


def test_something():
    assert True


def test_with_error():
    with pytest.raises(ValueError):
        # Do something that raises a ValueError
        raise(ValueError)


# Fixture example
@pytest.fixture
def an_object():
    return {}


def test_ml_lsmodel_ascat(an_object):
    assert an_object == {}
