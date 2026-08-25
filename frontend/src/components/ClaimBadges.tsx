import type { Claim } from "../api/types";

const VERDICT_CLASS: Record<Claim["verdict"], string> = {
  SUPPORTED: "verdict-pill--supported",
  CONTRADICTED: "verdict-pill--contradicted",
  UNVERIFIABLE: "verdict-pill--unverifiable"
};

export default function ClaimBadges({
  claim,
  showYearFlag = false
}: {
  claim: Claim;
  showYearFlag?: boolean;
}) {
  return (
    <>
      <span className={`verdict-pill ${VERDICT_CLASS[claim.verdict]}`}>{claim.verdict}</span>
      {claim.stance === "disavowed" ? (
        <span className="stance-chip" title="The report itself marks this claim false">
          disavowed by report
        </span>
      ) : null}
      {claim.downgraded ? (
        <span
          className="stance-chip"
          title="The quote check failed, so the verdict was downgraded to unverifiable"
        >
          downgraded
        </span>
      ) : null}
      {showYearFlag && claim.year_flag ? (
        <span
          className="stance-chip stance-chip--warning"
          title="The claim's year does not appear in the quoted evidence"
        >
          year mismatch
        </span>
      ) : null}
    </>
  );
}
