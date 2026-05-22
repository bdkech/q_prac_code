from loguru import logger


def gleason_to_grade_group(primary: int, secondary: int) -> int:
    """
    Map a Gleason score pair (primary + secondary) to ISUP grade group 1–5.

    The mapping follows the ISUP 2014/2016 consensus guidelines for prostate cancer
    grading. Higher grade groups indicate more aggressive cancer.

    Args:
        primary: Primary Gleason pattern (1–5)
        secondary: Secondary Gleason pattern (1–5)

    Returns:
        ISUP grade group (1–5)

    Raises:
        TypeError: If primary or secondary are not integers.
    """
    if not isinstance(primary, int) or not isinstance(secondary, int):
        error_msg = (
            f"Primary and secondary must be integers, got "
            f"primary={type(primary).__name__}, secondary={type(secondary).__name__}"
        )
        logger.error(error_msg)
        raise TypeError(error_msg)

    if primary < 1 or primary > 5:
        logger.warning(f"Primary Gleason pattern {primary} outside typical range (1-5)")
    if secondary < 1 or secondary > 5:
        logger.warning(
            f"Secondary Gleason pattern {secondary} outside typical range (1-5)"
        )

    gleason_sum = primary + secondary

    # Grade Group 1: Gleason ≤ 6 (3+3 or lower)
    if gleason_sum <= 6:
        return 1

    # Grade Group 2: Gleason 3+4 = 7
    if gleason_sum == 7 and primary == 3:
        return 2

    # Grade Group 3: Gleason 4+3 = 7
    if gleason_sum == 7 and primary == 4:
        return 3

    # Grade Group 4: Gleason 8
    if gleason_sum == 8:
        return 4

    # Grade Group 5: Gleason 9–10
    return 5


def parse_gs(
    gs_string: str | None,
) -> tuple[int | None, int | None, int | None, bool]:
    """
    Parse a comma-separated lesion Gleason score string.

    Processes entries like "3+4,3+3", "0+0", or null to extract maximum grade
    group, Gleason sum, and primary pattern across all lesions. The "0+0"
    sentinel value indicates no cancer detected.

    Args:
        gs_string: Comma-separated Gleason scores (e.g., "3+4,4+3") or None.

    Returns:
        Tuple of (grade_group_max, gleason_sum_max, gleason_primary_max, is_cspc):
        - grade_group_max: Maximum ISUP grade group across lesions (1–5 or None)
        - gleason_sum_max: Maximum Gleason sum across lesions (or None)
        - gleason_primary_max: Maximum primary Gleason pattern across lesions (or None)
        - is_cspc: True if clinically significant prostate cancer (grade group ≥ 2)

    Raises:
        TypeError: If input is not a string or None.
        ValueError: If a lesion entry contains non-numeric values.
    """
    if gs_string is not None and not isinstance(gs_string, str):
        error_msg = f"Input must be string or None, got {type(gs_string).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if not gs_string or gs_string.strip() == "":
        logger.warning("Empty or null Gleason score string")
        return (None, None, None, False)

    lesions = [lesion.strip() for lesion in gs_string.split(",")]

    grade_groups: list[int] = []
    gleason_sums: list[int] = []
    gleason_primaries: list[int] = []
    skipped_count = 0

    for lesion in lesions:
        if not lesion:
            logger.warning(f"Empty lesion entry in: {gs_string}")
            skipped_count += 1
            continue

        # "0+0" is the PI-CAI sentinel for a biopsy with no cancer found
        if lesion == "0+0":
            logger.warning(f"Found no-cancer sentinel (0+0) in: {gs_string}")
            continue

        parts = lesion.split("+")
        if len(parts) != 2:
            logger.warning(
                f"Malformed lesion entry '{lesion}' in {gs_string}: "
                f"expected 'X+Y' format, got {len(parts)} parts"
            )
            skipped_count += 1
            continue

        try:
            primary = int(parts[0].strip())
            secondary = int(parts[1].strip())
        except ValueError as e:
            error_msg = f"Non-numeric values in lesion '{lesion}' from {gs_string}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        try:
            grade_group = gleason_to_grade_group(primary, secondary)
            gleason_sum = primary + secondary
            grade_groups.append(grade_group)
            gleason_sums.append(gleason_sum)
            gleason_primaries.append(primary)
        except (TypeError, ValueError):
            logger.exception(f"Failed to process lesion '{lesion}'")
            raise

    if not grade_groups:
        if skipped_count > 0:
            logger.warning(
                f"No valid lesions found in {gs_string} "
                f"({skipped_count} entries skipped)"
            )
        else:
            logger.warning(f"No cancer found in: {gs_string}")
        return (None, None, None, False)

    grade_group_max = max(grade_groups)
    gleason_sum_max = max(gleason_sums)
    gleason_primary_max = max(gleason_primaries)

    # Clinically significant prostate cancer is defined as grade group ≥ 2
    is_cspc = grade_group_max >= 2

    return (grade_group_max, gleason_sum_max, gleason_primary_max, is_cspc)
