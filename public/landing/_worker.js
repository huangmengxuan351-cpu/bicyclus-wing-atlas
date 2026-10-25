// Lightweight project entry; the complete editions remain independent.
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (!['GET', 'HEAD'].includes(request.method)) return new Response('Method not allowed.', {status:405});
    if (url.pathname === '/__deployment_health') return Response.json({ready:true,service:'bicyclus-wing-atlas-entry',edition_sites:['en','zh']});
    if (url.pathname.startsWith('/downloads') || url.pathname.startsWith('/atlas-assets/') || url.pathname.startsWith('/anatomy-assets/')) return new Response('Original data require separate authorization on the language site.', {status:401});
    if (url.pathname === '/manual/') return Response.redirect('https://bicyclus-wing-atlas-en.pages.dev/manual/',302);
    for (const language of ['en','zh']) {
      if (url.pathname.startsWith('/'+language+'/')) {
        const path = url.pathname === '/'+language+'/manual/' ? '/manual/' : url.pathname;
        return Response.redirect('https://bicyclus-wing-atlas-'+language+'.pages.dev'+path+url.search,302);
      }
    }
    if (url.pathname === '/') return env.ASSETS.fetch(request);
    return new Response('Page not found.', {status:404});
  }
};
