"""Map SIC codes to the sector groups the synergy model is calibrated within.

Grouping is by cost *structure*, not by GICS-style end market, because that is
what determines how much overlap a merger can actually remove. The model is
linear in SG&A and COGS, so what matters is whether a sector's costs sit in
duplicated corporate overhead (removable) or in per-unit operations
(not removable). Grocery retail and enterprise software can both be "consumer-
facing" and behave nothing alike here: Kroger's SG&A is store-level labor that
survives a merger, Synopsys's is corporate function that does not.

Sectors flagged `supported=False` report costs by nature rather than by
function -- they have no comparable COGS/SG&A split at all, so the model's
framing does not apply and their deals are excluded from calibration rather
than fitted badly. This mirrors the exclusions already documented in
`validation/deals.py`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Sector:
    key: str
    label: str
    supported: bool = True
    note: str | None = None


_UNKNOWN = Sector("unknown", "Unknown / unmapped SIC")

# Keyed by SIC major group (the first two digits of the 4-digit code).
_MAJOR_GROUPS: dict[str, Sector] = {
    # -- Structurally out of scope -------------------------------------------
    "60": Sector("banking", "Depository institutions", False, "no COGS/SG&A split"),
    "61": Sector("banking", "Non-depository credit", False, "no COGS/SG&A split"),
    "62": Sector("banking", "Security & commodity brokers", False, "no COGS/SG&A split"),
    "63": Sector("insurance", "Insurance carriers", False, "premiums/reserves, not COGS"),
    "64": Sector("insurance", "Insurance agents", False, "premiums/reserves, not COGS"),
    "65": Sector("real_estate", "Real estate", False, "property-level costs, not COGS"),
    "67": Sector("holding", "Holding & investment offices", False, "no operating cost structure"),
    "13": Sector("energy_upstream", "Oil & gas extraction", False, "cost-by-nature: production, DD&A"),
    "29": Sector("energy_refining", "Petroleum refining", False, "cost-by-nature: feedstock"),
    "45": Sector("airlines", "Air transportation", False, "cost-by-nature: fuel, labor, aircraft rent"),
    # -- Supported ------------------------------------------------------------
    "73": Sector("software_services", "Business services & software"),
    "737": Sector("software_services", "Computer & software services"),
    "36": Sector("electronics", "Electronic & electrical equipment"),
    "38": Sector("instruments", "Instruments & medical devices"),
    "35": Sector("industrial_machinery", "Industrial & commercial machinery"),
    "28": Sector("pharma_chemicals", "Chemicals & pharmaceuticals"),
    "20": Sector("food_beverage", "Food & kindred products"),
    "26": Sector("paper_packaging", "Paper & allied products"),
    "30": Sector("rubber_plastics", "Rubber & plastics"),
    "33": Sector("metals", "Primary metals"),
    "34": Sector("fabricated_metals", "Fabricated metal products"),
    "37": Sector("transport_equipment", "Transportation equipment"),
    "48": Sector("telecom", "Communications"),
    "49": Sector("utilities", "Electric, gas & sanitary services"),
    "50": Sector("wholesale", "Wholesale trade - durable goods"),
    "51": Sector("wholesale", "Wholesale trade - nondurable goods"),
    "53": Sector("retail", "General merchandise retail"),
    "54": Sector("retail_grocery", "Food stores"),
    "55": Sector("retail", "Automotive dealers"),
    "56": Sector("retail", "Apparel & accessory stores"),
    "57": Sector("retail", "Furniture & home furnishings stores"),
    "58": Sector("restaurants", "Eating & drinking places"),
    "59": Sector("retail", "Miscellaneous retail"),
    "80": Sector("healthcare_services", "Health services"),
    "87": Sector("professional_services", "Engineering & management services"),
    "42": Sector("logistics", "Motor freight & warehousing"),
    "47": Sector("logistics", "Transportation services"),
    "79": Sector("leisure", "Amusement & recreation services"),
    "70": Sector("lodging", "Hotels & lodging"),
    "23": Sector("apparel", "Apparel & textile products"),
    "24": Sector("building_products", "Lumber & wood products"),
    "32": Sector("building_products", "Stone, clay & glass"),
    "16": Sector("construction", "Heavy construction"),
    "17": Sector("construction", "Special trade contractors"),
    "27": Sector("media_publishing", "Printing & publishing"),
    "78": Sector("media_publishing", "Motion pictures"),
    "22": Sector("textiles", "Textile mill products"),
    "25": Sector("furniture", "Furniture & fixtures"),
    "39": Sector("misc_manufacturing", "Miscellaneous manufacturing"),
    "72": Sector("consumer_services", "Personal services"),
    "76": Sector("consumer_services", "Repair services"),
    "82": Sector("education", "Educational services"),
    "83": Sector("social_services", "Social services"),
}


def sector_for_sic(sic: str | int | None) -> Sector:
    """Resolve a 4-digit SIC code to its calibration sector.

    Falls back from the 3-digit prefix to the 2-digit major group, so a code
    with a more specific override (e.g. 737x software) picks that up first.
    """
    if sic is None:
        return _UNKNOWN
    code = str(sic).strip().lstrip("0") or "0"
    code = code.zfill(4) if len(code) < 4 else code
    return _MAJOR_GROUPS.get(code[:3]) or _MAJOR_GROUPS.get(code[:2]) or _UNKNOWN


def is_supported(sic: str | int | None) -> bool:
    """Whether the model's cost/COGS framing applies to this filer at all."""
    return sector_for_sic(sic).supported
