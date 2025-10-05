/*
    @Positive
 * Copyright (c) 2001, 2019, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.net.URL;
    @Positive
import java.net.Proxy;
    @Positive
import java.net.ProtocolException;
    @Positive
import java.net.MalformedURLException;
    @Positive
import java.io.*;
    @Positive
import java.net.Authenticator;
    @Positive
import javax.net.ssl.*;
    @Positive
import java.security.Permission;
    @Positive
import java.security.Principal;
    @Positive
import java.util.Map;
    @Positive
import java.util.List;
    @Positive
import java.util.Optional;
    @Positive
import sun.net.util.IPAddressUtil;
    @Positive
import sun.net.www.http.HttpClient;

    @Positive
public class HttpsURLConnectionImpl extends javax.net.ssl.HttpsURLConnection {

    @Positive
    static URL checkURL(URL u) throws IOException;

    @Positive
    protected void setNewClient(URL url) throws IOException;

    @Positive
    protected void setNewClient(URL url, boolean useCache) throws IOException;

    @Positive
    protected void setProxiedClient(URL url, String proxyHost, int proxyPort) throws IOException;

    @Positive
    protected void setProxiedClient(URL url, String proxyHost, int proxyPort, boolean useCache) throws IOException;

    @Positive
    public void connect() throws IOException;

    @Positive
    protected boolean isConnected();

    @Positive
    protected void setConnected(boolean conn);

    @Positive
    public String getCipherSuite();

    @Positive
    public java.security.cert.Certificate[] getLocalCertificates();

    @Positive
    public java.security.cert.Certificate[] getServerCertificates() throws SSLPeerUnverifiedException;

    @Positive
    public Principal getPeerPrincipal() throws SSLPeerUnverifiedException;

    @Positive
    public Principal getLocalPrincipal();

    @Positive
    public OutputStream getOutputStream() throws IOException;

    @Positive
    public InputStream getInputStream() throws IOException;

    @Positive
    public InputStream getErrorStream();

    @Positive
    public void disconnect();

    @Positive
    public boolean usingProxy();

    @Positive
    public Map<String, List<String>> getHeaderFields();

    @Positive
    public String getHeaderField(String name);

    @Positive
    public String getHeaderField(int n);

    @Positive
    public String getHeaderFieldKey(int n);

    @Positive
    public void setRequestProperty(String key, String value);

    @Positive
    public void addRequestProperty(String key, String value);

    @Positive
    public int getResponseCode() throws IOException;

    @Positive
    public String getRequestProperty(String key);

    @Positive
    public Map<String, List<String>> getRequestProperties();

    @Positive
    public void setInstanceFollowRedirects(boolean shouldFollow);

    @Positive
    public boolean getInstanceFollowRedirects();

    @Positive
    public void setRequestMethod(String method) throws ProtocolException;

    @Positive
    public String getRequestMethod();

    @Positive
    public String getResponseMessage() throws IOException;

    @Positive
    public long getHeaderFieldDate(String name, long Default);

    @Positive
    public Permission getPermission() throws IOException;

    @Positive
    public URL getURL();

    @Positive
    public int getContentLength();

    @Positive
    public long getContentLengthLong();

    @Positive
    public String getContentType();

    @Positive
    public String getContentEncoding();

    @Positive
    public long getExpiration();

    @Positive
    public long getDate();

    @Positive
    public long getLastModified();

    @Positive
    public int getHeaderFieldInt(String name, int Default);

    @Positive
    public long getHeaderFieldLong(String name, long Default);

    @Positive
    public Object getContent() throws IOException;

    @Positive
    @SuppressWarnings("rawtypes")
    @Positive
    public Object getContent(Class[] classes) throws IOException;

    @Positive
    public String toString();

    @Positive
    public void setDoInput(boolean doinput);

    @Positive
    public boolean getDoInput();

    @Positive
    public void setDoOutput(boolean dooutput);

    @Positive
    public boolean getDoOutput();

    @Positive
    public void setAllowUserInteraction(boolean allowuserinteraction);

    @Positive
    public boolean getAllowUserInteraction();

    @Positive
    public void setUseCaches(boolean usecaches);

    @Positive
    public boolean getUseCaches();

    @Positive
    public void setIfModifiedSince(long ifmodifiedsince);

    @Positive
    public long getIfModifiedSince();

    @Positive
    public boolean getDefaultUseCaches();

    @Positive
    public void setDefaultUseCaches(boolean defaultusecaches);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public void setConnectTimeout(int timeout);

    @Positive
    public int getConnectTimeout();

    @Positive
    public void setReadTimeout(int timeout);

    @Positive
    public int getReadTimeout();

    @Positive
    public void setFixedLengthStreamingMode(int contentLength);

    @Positive
    public void setFixedLengthStreamingMode(long contentLength);

    @Positive
    public void setChunkedStreamingMode(int chunklen);

    @Positive
    @Override
    @Positive
    public void setAuthenticator(Authenticator auth);

    @Positive
    @Override
    @Positive
    public Optional<SSLSession> getSSLSession();
    @Positive
}

// CFWR semantic augmentation - variant 0
