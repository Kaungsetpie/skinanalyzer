const baseUrl = "http://192.168.100.235:8000";

export async function apiRequest(
  endpoint: string,
  method: string = "GET",
  data?: any,
  type?: string,
) {
  const url = `${baseUrl}/${endpoint}`;
  console.log(
    `${type} Making API request to: ${url} with method: ${method} and data:`,
    data,
  );
  const isFormData = type === "formData";
  const options: RequestInit = {
    method,
    headers: {
      accept: "application/json",
      ...(data !== undefined && !isFormData ? { "Content-Type": "application/json" } : {}),
    },
  };
  if (isFormData) {
    options.body = data;
  } else if (data !== undefined) {
    options.body = JSON.stringify(data);
  }
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail: string | undefined;
      try {
        const body = await response.json();
        detail = body?.detail;
      } catch {}
      throw Object.assign(new Error(detail ?? `HTTP error! status: ${response.status}`), {
        status: response.status,
      });
    }
    return await response.json();
  } catch (error) {
    console.error("API request error:", error);
    throw error;
  }
}
