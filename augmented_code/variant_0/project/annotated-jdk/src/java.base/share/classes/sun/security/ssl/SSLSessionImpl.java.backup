/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.ssl;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import sun.security.x509.X509CertImpl;
import java.io.IOException;
import java.math.BigInteger;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.security.Principal;
import java.security.PrivateKey;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Queue;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.locks.ReentrantLock;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import javax.net.ssl.ExtendedSSLSession;
import javax.net.ssl.SNIHostName;
import javax.net.ssl.SNIServerName;
import javax.net.ssl.SSLException;
import javax.net.ssl.SSLPeerUnverifiedException;
import javax.net.ssl.SSLPermission;
import javax.net.ssl.SSLSessionBindingEvent;
import javax.net.ssl.SSLSessionBindingListener;
import javax.net.ssl.SSLSessionContext;

final class SSLSessionImpl extends ExtendedSSLSession {

    boolean isStatelessable();

    byte[] write() throws Exception;

    void setMasterSecret(SecretKey secret);

    void setResumptionMasterSecret(SecretKey secret);

    void setPreSharedKey(SecretKey key);

    void addChild(SSLSessionImpl session);

    void setTicketAgeAdd(int ticketAgeAdd);

    void setPskIdentity(byte[] pskIdentity);

    BigInteger incrTicketNonceCounter();

    boolean isPSKable();

    SecretKey getMasterSecret();

    SecretKey getResumptionMasterSecret();

    SecretKey getPreSharedKey();

    SecretKey consumePreSharedKey();

    int getTicketAgeAdd();

    String getIdentificationProtocol();

    byte[] consumePskIdentity();

    byte[] getPskIdentity();

    void setPeerCertificates(X509Certificate[] peer);

    void setLocalCertificates(X509Certificate[] local);

    void setLocalPrivateKey(PrivateKey privateKey);

    void setPeerSupportedSignatureAlgorithms(Collection<SignatureScheme> signatureSchemes);

    void setUseDefaultPeerSignAlgs();

    SSLSessionImpl finish();

    void setStatusResponses(List<byte[]> responses);

    boolean isRejoinable();

    @Override
    public boolean isValid();

    @Override
    public byte[] getId();

    @SuppressWarnings("removal")
    @Override
    public SSLSessionContext getSessionContext();

    SessionId getSessionId();

    CipherSuite getSuite();

    void setSuite(CipherSuite suite);

    boolean isSessionResumption();

    void setAsSessionResumption(boolean flag);

    @Override
    public String getCipherSuite();

    ProtocolVersion getProtocolVersion();

    @Override
    public String getProtocol();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public java.security.cert.Certificate[] getPeerCertificates() throws SSLPeerUnverifiedException;

    @Override
    public java.security.cert.Certificate[] getLocalCertificates();

    public X509Certificate[] getCertificateChain() throws SSLPeerUnverifiedException;

    @Override
    public List<byte[]> getStatusResponses();

    @Override
    public Principal getPeerPrincipal() throws SSLPeerUnverifiedException;

    @Override
    public Principal getLocalPrincipal();

    public long getTicketCreationTime();

    @Override
    public long getCreationTime();

    @Override
    public long getLastAccessedTime();

    void setLastAccessedTime(long time);

    public InetAddress getPeerAddress();

    @Override
    public String getPeerHost();

    @Override
    public int getPeerPort();

    void setContext(SSLSessionContextImpl ctx);

    @Override
    public void invalidate();

    @Override
    public void putValue(String key, Object value);

    @Override
    public Object getValue(String key);

    @Override
    public void removeValue(String key);

    @Override
    public String[] getValueNames();

    protected void expandBufferSizes();

    @Override
    public int getPacketBufferSize();

    @Override
    public int getApplicationBufferSize();

    void setNegotiatedMaxFragSize(int negotiatedMaxFragLen);

    int getNegotiatedMaxFragSize();

    void setMaximumPacketSize(int maximumPacketSize);

    int getMaximumPacketSize();

    @Override
    public String[] getLocalSupportedSignatureAlgorithms();

    public Collection<SignatureScheme> getLocalSupportedSignatureSchemes();

    @Override
    public String[] getPeerSupportedSignatureAlgorithms();

    @Override
    public List<SNIServerName> getRequestedServerNames();

    @Override
    public String toString();
}

class SecureKey {

    static Object getCurrentSecurityContext();

    Object getAppKey();

    Object getSecurityContext();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);
}
