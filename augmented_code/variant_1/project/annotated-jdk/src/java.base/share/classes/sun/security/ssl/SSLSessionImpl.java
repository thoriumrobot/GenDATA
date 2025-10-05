/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.ssl;

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
import sun.security.x509.X509CertImpl;
    @Positive
import java.io.IOException;
    @Positive
import java.math.BigInteger;
    @Positive
import java.net.InetAddress;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.security.Principal;
    @Positive
import java.security.PrivateKey;
    @Positive
import java.security.cert.X509Certificate;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Queue;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.List;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentLinkedQueue;
    @Positive
import java.util.concurrent.locks.ReentrantLock;
    @Positive
import javax.crypto.SecretKey;
    @Positive
import javax.crypto.spec.SecretKeySpec;
    @Positive
import javax.net.ssl.ExtendedSSLSession;
    @Positive
import javax.net.ssl.SNIHostName;
    @Positive
import javax.net.ssl.SNIServerName;
    @Positive
import javax.net.ssl.SSLException;
    @Positive
import javax.net.ssl.SSLPeerUnverifiedException;
    @Positive
import javax.net.ssl.SSLPermission;
    @Positive
import javax.net.ssl.SSLSessionBindingEvent;
    @Positive
import javax.net.ssl.SSLSessionBindingListener;
    @Positive
import javax.net.ssl.SSLSessionContext;

    @Positive
final class SSLSessionImpl extends ExtendedSSLSession {

    @Positive
    boolean isStatelessable();

    @Positive
    byte[] write() throws Exception;

    @Positive
    void setMasterSecret(SecretKey secret);

    @Positive
    void setResumptionMasterSecret(SecretKey secret);

    @Positive
    void setPreSharedKey(SecretKey key);

    @Positive
    void addChild(SSLSessionImpl session);

    @Positive
    void setTicketAgeAdd(int ticketAgeAdd);

    @Positive
    void setPskIdentity(byte[] pskIdentity);

    @Positive
    BigInteger incrTicketNonceCounter();

    @Positive
    boolean isPSKable();

    @Positive
    SecretKey getMasterSecret();

    @Positive
    SecretKey getResumptionMasterSecret();

    @Positive
    SecretKey getPreSharedKey();

    @Positive
    SecretKey consumePreSharedKey();

    @Positive
    int getTicketAgeAdd();

    @Positive
    String getIdentificationProtocol();

    @Positive
    byte[] consumePskIdentity();

    @Positive
    byte[] getPskIdentity();

    @Positive
    void setPeerCertificates(X509Certificate[] peer);

    @Positive
    void setLocalCertificates(X509Certificate[] local);

    @Positive
    void setLocalPrivateKey(PrivateKey privateKey);

    @Positive
    void setPeerSupportedSignatureAlgorithms(Collection<SignatureScheme> signatureSchemes);

    @Positive
    void setUseDefaultPeerSignAlgs();

    @Positive
    SSLSessionImpl finish();

    @Positive
    void setStatusResponses(List<byte[]> responses);

    @Positive
    boolean isRejoinable();

    @Positive
    @Override
    @Positive
    public boolean isValid();

    @Positive
    @Override
    @Positive
    public byte[] getId();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @Override
    @Positive
    public SSLSessionContext getSessionContext();

    @Positive
    SessionId getSessionId();

    @Positive
    CipherSuite getSuite();

    @Positive
    void setSuite(CipherSuite suite);

    @Positive
    boolean isSessionResumption();

    @Positive
    void setAsSessionResumption(boolean flag);

    @Positive
    @Override
    @Positive
    public String getCipherSuite();

    @Positive
    ProtocolVersion getProtocolVersion();

    @Positive
    @Override
    @Positive
    public String getProtocol();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public java.security.cert.Certificate[] getPeerCertificates() throws SSLPeerUnverifiedException;

    @Positive
    @Override
    @Positive
    public java.security.cert.Certificate[] getLocalCertificates();

    @Positive
    public X509Certificate[] getCertificateChain() throws SSLPeerUnverifiedException;

    @Positive
    @Override
    @Positive
    public List<byte[]> getStatusResponses();

    @Positive
    @Override
    @Positive
    public Principal getPeerPrincipal() throws SSLPeerUnverifiedException;

    @Positive
    @Override
    @Positive
    public Principal getLocalPrincipal();

    @Positive
    public long getTicketCreationTime();

    @Positive
    @Override
    @Positive
    public long getCreationTime();

    @Positive
    @Override
    @Positive
    public long getLastAccessedTime();

    @Positive
    void setLastAccessedTime(long time);

    @Positive
    public InetAddress getPeerAddress();

    @Positive
    @Override
    @Positive
    public String getPeerHost();

    @Positive
    @Override
    @Positive
    public int getPeerPort();

    @Positive
    void setContext(SSLSessionContextImpl ctx);

    @Positive
    @Override
    @Positive
    public void invalidate();

    @Positive
    @Override
    @Positive
    public void putValue(String key, Object value);

    @Positive
    @Override
    @Positive
    public Object getValue(String key);

    @Positive
    @Override
    @Positive
    public void removeValue(String key);

    @Positive
    @Override
    @Positive
    public String[] getValueNames();

    @Positive
    protected void expandBufferSizes();

    @Positive
    @Override
    @Positive
    public int getPacketBufferSize();

    @Positive
    @Override
    @Positive
    public int getApplicationBufferSize();

    @Positive
    void setNegotiatedMaxFragSize(int negotiatedMaxFragLen);

    @Positive
    int getNegotiatedMaxFragSize();

    @Positive
    void setMaximumPacketSize(int maximumPacketSize);

    @Positive
    int getMaximumPacketSize();

    @Positive
    @Override
    @Positive
    public String[] getLocalSupportedSignatureAlgorithms();

    @Positive
    public Collection<SignatureScheme> getLocalSupportedSignatureSchemes();

    @Positive
    @Override
    @Positive
    public String[] getPeerSupportedSignatureAlgorithms();

    @Positive
    @Override
    @Positive
    public List<SNIServerName> getRequestedServerNames();

    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}

    @Positive
class SecureKey {

    @Positive
    static Object getCurrentSecurityContext();

    @Positive
    Object getAppKey();

    @Positive
    Object getSecurityContext();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);
    @Positive
}
