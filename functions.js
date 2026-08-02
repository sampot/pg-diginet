/** Optional Playgrounds Infrastructure stub. */
export default {
  async fetch(request) {
    return Response.json({
      ok: true,
      name: "pg-diginet",
      path: new URL(request.url).pathname,
    });
  },
};
