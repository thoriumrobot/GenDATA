/*
 * Copyright (c) 2001, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package sun.net.www.protocol.https;

import org.checkerframework.checker.signature.qual.CanonicalName;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.io.PrintStream;
import java.io.BufferedOutputStream;
import java.net.InetAddress;
import java.net.Socket;
import java.net.SocketException;
import java.net.URL;
import java.net.UnknownHostException;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.security.Principal;
import java.security.cert.*;
import java.util.List;
import java.util.Objects;
import java.util.StringTokenizer;
import java.util.Vector;
import javax.security.auth.x500.X500Principal;
import javax.net.ssl.*;
import sun.net.www.http.HttpClient;
import sun.net.www.protocol.http.AuthenticatorKeys;
import sun.net.www.protocol.http.HttpURLConnection;
import sun.security.action.*;
import sun.security.util.HostnameChecker;
import sun.security.ssl.SSLSocketImpl;
import sun.util.logging.PlatformLogger;
import static sun.net.www.protocol.http.HttpURLConnection.TunnelState.*;

final class HttpsClient extends HttpClient implements HandshakeCompletedListener {

    @Override
    protected int getDefaultPort();

    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, HttpURLConnection httpuc) throws IOException;

    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, boolean useCache, HttpURLConnection httpuc) throws IOException;

    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, String proxyHost, int proxyPort, HttpURLConnection httpuc) throws IOException;

    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, String proxyHost, int proxyPort, boolean useCache, HttpURLConnection httpuc) throws IOException;

    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, String proxyHost, int proxyPort, boolean useCache, int connectTimeout, HttpURLConnection httpuc) throws IOException;

    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, Proxy p, boolean useCache, int connectTimeout, HttpURLConnection httpuc) throws IOException;

    void setHostnameVerifier(HostnameVerifier hv);

    void setSSLSocketFactory(SSLSocketFactory sf);

    SSLSocketFactory getSSLSocketFactory();

    @Override
    protected Socket createSocket() throws IOException;

    @Override
    public boolean needsTunneling();

    @Override
    public void afterConnect() throws IOException, UnknownHostException;

    @Override
    protected void putInKeepAliveCache();

    @Override
    public void closeIdleConnection();

    String getCipherSuite();

    public java.security.cert.Certificate[] getLocalCertificates();

    java.security.cert.Certificate[] getServerCertificates() throws SSLPeerUnverifiedException;

    Principal getPeerPrincipal() throws SSLPeerUnverifiedException;

    Principal getLocalPrincipal();

    SSLSession getSSLSession();

    public void handshakeCompleted(HandshakeCompletedEvent event);

    @Override
    public String getProxyHostUsed();

    @Override
    public int getProxyPortUsed();
}
