/*
    @Positive
 * Copyright (c) 2001, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package sun.net.www.protocol.https;

    @Positive
import org.checkerframework.checker.signature.qual.CanonicalName;
    @Positive
import java.io.IOException;
    @Positive
import java.io.UnsupportedEncodingException;
    @Positive
import java.io.PrintStream;
    @Positive
import java.io.BufferedOutputStream;
    @Positive
import java.net.InetAddress;
    @Positive
import java.net.Socket;
    @Positive
import java.net.SocketException;
    @Positive
import java.net.URL;
    @Positive
import java.net.UnknownHostException;
    @Positive
import java.net.InetSocketAddress;
    @Positive
import java.net.Proxy;
    @Positive
import java.security.Principal;
    @Positive
import java.security.cert.*;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;
    @Positive
import java.util.StringTokenizer;
    @Positive
import java.util.Vector;
    @Positive
import javax.security.auth.x500.X500Principal;
    @Positive
import javax.net.ssl.*;
    @Positive
import sun.net.www.http.HttpClient;
    @Positive
import sun.net.www.protocol.http.AuthenticatorKeys;
    @Positive
import sun.net.www.protocol.http.HttpURLConnection;
    @Positive
import sun.security.action.*;
    @Positive
import sun.security.util.HostnameChecker;
    @Positive
import sun.security.ssl.SSLSocketImpl;
    @Positive
import sun.util.logging.PlatformLogger;
    @Positive
import static sun.net.www.protocol.http.HttpURLConnection.TunnelState.*;

    @Positive
final class HttpsClient extends HttpClient implements HandshakeCompletedListener {

    @Positive
    @Override
    @Positive
    protected int getDefaultPort();

    @Positive
    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, HttpURLConnection httpuc) throws IOException;

    @Positive
    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, boolean useCache, HttpURLConnection httpuc) throws IOException;

    @Positive
    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, String proxyHost, int proxyPort, HttpURLConnection httpuc) throws IOException;

    @Positive
    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, String proxyHost, int proxyPort, boolean useCache, HttpURLConnection httpuc) throws IOException;

    @Positive
    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, String proxyHost, int proxyPort, boolean useCache, int connectTimeout, HttpURLConnection httpuc) throws IOException;

    @Positive
    static HttpClient New(SSLSocketFactory sf, URL url, HostnameVerifier hv, Proxy p, boolean useCache, int connectTimeout, HttpURLConnection httpuc) throws IOException;

    @Positive
    void setHostnameVerifier(HostnameVerifier hv);

    @Positive
    void setSSLSocketFactory(SSLSocketFactory sf);

    @Positive
    SSLSocketFactory getSSLSocketFactory();

    @Positive
    @Override
    @Positive
    protected Socket createSocket() throws IOException;

    @Positive
    @Override
    @Positive
    public boolean needsTunneling();

    @Positive
    @Override
    @Positive
    public void afterConnect() throws IOException, UnknownHostException;

    @Positive
    @Override
    @Positive
    protected void putInKeepAliveCache();

    @Positive
    @Override
    @Positive
    public void closeIdleConnection();

    @Positive
    String getCipherSuite();

    @Positive
    public java.security.cert.Certificate[] getLocalCertificates();

    @Positive
    java.security.cert.Certificate[] getServerCertificates() throws SSLPeerUnverifiedException;

    @Positive
    Principal getPeerPrincipal() throws SSLPeerUnverifiedException;

    @Positive
    Principal getLocalPrincipal();

    @Positive
    SSLSession getSSLSession();

    @Positive
    public void handshakeCompleted(HandshakeCompletedEvent event);

    @Positive
    @Override
    @Positive
    public String getProxyHostUsed();

    @Positive
    @Override
    @Positive
    public int getProxyPortUsed();
    @Positive
}

// CFWR semantic augmentation - variant 0
