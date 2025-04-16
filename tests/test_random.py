import iwutil
import numpy as np
from time import sleep


def test_current_time_integer():
    # Test that the function returns an integer
    result = iwutil.random.current_time_integer()
    assert isinstance(result, int)

    # Test that the result has the expected format (%Y%m%d%H%M%S%f)
    # Should be at least 14 digits (%Y%m%d%H%M%S%f) but may include microseconds
    assert len(str(result)) >= 14

    # Test that two consecutive calls give different results
    result1 = iwutil.random.current_time_integer()
    sleep(0.01)  # Sleep to ensure different time-based seeds
    result2 = iwutil.random.current_time_integer()
    assert result2 > result1


def test_seed():
    # Test setting a specific seed
    specific_seed = 12345
    iwutil.random.seed(specific_seed)
    assert iwutil.random.get_seed() == specific_seed

    # Generate some random numbers with this seed
    random_numbers1 = np.random.rand(5)

    # Reset to the same seed and verify we get the same numbers
    iwutil.random.seed(specific_seed)
    random_numbers2 = np.random.rand(5)
    np.testing.assert_array_equal(random_numbers1, random_numbers2)

    # Test setting seed with no value (should use current time)
    iwutil.random.seed()
    seed1 = iwutil.random.get_seed()

    # Verify that automatic seeds are within valid range for numpy
    # (signed 32-bit integer)
    assert seed1 >= 0 and seed1 < 2**32

    # Test that two automatic seeds are different
    sleep(0.01)  # Sleep to ensure different time-based seeds
    iwutil.random.seed()
    seed2 = iwutil.random.get_seed()
    assert seed1 != seed2


def test_generate_seed():
    # Test that the function returns an integer
    result = iwutil.random.generate_seed()
    sleep(0.01)
    assert isinstance(result, int)

    # Test that the result is within valid range for numpy seeds (0 to 2^32)
    assert result >= 0
    assert result < 2**32

    # Test that two consecutive calls give different results
    result1 = iwutil.random.generate_seed()
    sleep(0.01)  # Sleep to ensure different time-based seeds
    result2 = iwutil.random.generate_seed()
    sleep(0.01)
    assert result1 != result2

    # Test that the seed is derived from current time
    # Get two time readings with a seed in between

    time1 = iwutil.random.current_time_integer()
    sleep(0.01)
    seed = iwutil.random.generate_seed()
    sleep(0.01)
    time2 = iwutil.random.current_time_integer()

    # The seed should be between time1 and time2 (modulo 2^32)
    seed_time = seed
    time1_mod = time1 % 2**32
    time2_mod = time2 % 2**32

    # If time2 is less than time1 after modulo, it means we wrapped around
    # In this case, a valid seed could be:
    # 1. Greater than or equal to time1_mod (up to 2^32)
    # 2. Less than or equal to time2_mod (starting from 0)
    if time2_mod < time1_mod:
        assert (
            seed_time >= time1_mod or seed_time <= time2_mod
        ), f"Seed {seed_time} should be >= {time1_mod} or <= {time2_mod} in wraparound case"
    else:
        assert (
            time1_mod <= seed_time <= time2_mod
        ), f"Seed {seed_time} should be between {time1_mod} and {time2_mod}"
