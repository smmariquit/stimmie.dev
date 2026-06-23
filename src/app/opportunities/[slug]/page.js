import { redirect } from "next/navigation";

/** Legacy issue URLs → single opportunities board. */
export default function LegacyOpportunityIssueRedirect() {
  redirect("/opportunities");
}
