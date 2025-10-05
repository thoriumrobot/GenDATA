/*
 * Copyright (c) 2001, 2019, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.net.URL;
import java.net.Proxy;
import java.net.ProtocolException;
import java.net.MalformedURLException;
import java.io.*;
import java.net.Authenticator;
import javax.net.ssl.*;
import java.security.Permission;
import java.security.Principal;
import java.util.Map;
import java.util.List;
import java.util.Optional;
import sun.net.util.IPAddressUtil;
import sun.net.www.http.HttpClient;

public class HttpsURLConnectionImpl extends javax.net.ssl.HttpsURLConnection {

    static URL checkURL(URL u) throws IOException;

    protected void setNewClient(URL url) throws IOException;

    protected void setNewClient(URL url, boolean useCache) throws IOException;

    protected void setProxiedClient(URL url, String proxyHost, int proxyPort) throws IOException;

    protected void setProxiedClient(URL url, String proxyHost, int proxyPort, boolean useCache) throws IOException;

    public void connect() throws IOException;

    protected boolean isConnected();

    protected void setConnected(boolean conn);

    public String getCipherSuite();

    public java.security.cert.Certificate[] getLocalCertificates();

    public java.security.cert.Certificate[] getServerCertificates() throws SSLPeerUnverifiedException;

    public Principal getPeerPrincipal() throws SSLPeerUnverifiedException;

    public Principal getLocalPrincipal();

    public OutputStream getOutputStream() throws IOException;

    public InputStream getInputStream() throws IOException;

    public InputStream getErrorStream();

    public void disconnect();

    public boolean usingProxy();

    public Map<String, List<String>> getHeaderFields();

    public String getHeaderField(String name);

    public String getHeaderField(int n);

    public String getHeaderFieldKey(int n);

    public void setRequestProperty(String key, String value);

    public void addRequestProperty(String key, String value);

    public int getResponseCode() throws IOException;

    public String getRequestProperty(String key);

    public Map<String, List<String>> getRequestProperties();

    public void setInstanceFollowRedirects(boolean shouldFollow);

    public boolean getInstanceFollowRedirects();

    public void setRequestMethod(String method) throws ProtocolException;

    public String getRequestMethod();

    public String getResponseMessage() throws IOException;

    public long getHeaderFieldDate(String name, long Default);

    public Permission getPermission() throws IOException;

    public URL getURL();

    public int getContentLength();

    public long getContentLengthLong();

    public String getContentType();

    public String getContentEncoding();

    public long getExpiration();

    public long getDate();

    public long getLastModified();

    public int getHeaderFieldInt(String name, int Default);

    public long getHeaderFieldLong(String name, long Default);

    public Object getContent() throws IOException;

    @SuppressWarnings("rawtypes")
    public Object getContent(Class[] classes) throws IOException;

    public String toString();

    public void setDoInput(boolean doinput);

    public boolean getDoInput();

    public void setDoOutput(boolean dooutput);

    public boolean getDoOutput();

    public void setAllowUserInteraction(boolean allowuserinteraction);

    public boolean getAllowUserInteraction();

    public void setUseCaches(boolean usecaches);

    public boolean getUseCaches();

    public void setIfModifiedSince(long ifmodifiedsince);

    public long getIfModifiedSince();

    public boolean getDefaultUseCaches();

    public void setDefaultUseCaches(boolean defaultusecaches);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public void setConnectTimeout(int timeout);

    public int getConnectTimeout();

    public void setReadTimeout(int timeout);

    public int getReadTimeout();

    public void setFixedLengthStreamingMode(int contentLength);

    public void setFixedLengthStreamingMode(long contentLength);

    public void setChunkedStreamingMode(int chunklen);

    @Override
    public void setAuthenticator(Authenticator auth);

    @Override
    public Optional<SSLSession> getSSLSession();
}
