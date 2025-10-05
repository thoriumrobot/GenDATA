/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.pkcs11;

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
import java.io.*;
    @Positive
import java.lang.ref.*;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.*;
    @Positive
import java.security.*;
    @Positive
import java.security.interfaces.*;
    @Positive
import java.security.spec.*;
    @Positive
import javax.crypto.*;
    @Positive
import javax.crypto.interfaces.*;
    @Positive
import javax.crypto.spec.*;
    @Positive
import sun.security.rsa.RSAUtil.KeyType;
    @Positive
import sun.security.rsa.RSAPublicKeyImpl;
    @Positive
import sun.security.rsa.RSAPrivateCrtKeyImpl;
    @Positive
import sun.security.internal.interfaces.TlsMasterSecret;
    @Positive
import sun.security.pkcs11.wrapper.*;
    @Positive
import static sun.security.pkcs11.TemplateManager.O_GENERATE;
    @Positive
import static sun.security.pkcs11.wrapper.PKCS11Constants.*;
    @Positive
import sun.security.util.DerValue;
    @Positive
import sun.security.util.Length;
    @Positive
import sun.security.util.ECUtil;
    @Positive
import sun.security.jca.JCAUtil;

    @Positive
abstract class P11Key implements Key, Length {

    @Positive
    public long getKeyID();

    @Positive
    public void releaseKeyID();

    @Positive
    public final String getAlgorithm();

    @Positive
    public final byte[] getEncoded();

    @Positive
    static boolean drainRefQueue();

    @Positive
    abstract byte[] getEncodedInternal();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    protected Object writeReplace() throws ObjectStreamException;

    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public int length();

    @Positive
    boolean isPublic();

    @Positive
    boolean isPrivate();

    @Positive
    boolean isSecret();

    @Positive
    void fetchAttributes(CK_ATTRIBUTE[] attributes);

    @Positive
    static SecretKey secretKey(Session session, long keyID, String algorithm, int keyLength, CK_ATTRIBUTE[] attributes);

    @Positive
    static SecretKey masterSecretKey(Session session, long keyID, String algorithm, int keyLength, CK_ATTRIBUTE[] attributes, int major, int minor);

    @Positive
    static PublicKey publicKey(Session session, long keyID, String algorithm, int keyLength, CK_ATTRIBUTE[] attributes);

    @Positive
    static PrivateKey privateKey(Session session, long keyID, String algorithm, int keyLength, CK_ATTRIBUTE[] attributes);

    @Positive
    private static final class P11PrivateKey extends P11Key implements PrivateKey {

    @Positive
        public String getFormat();

    @Positive
        byte[] getEncodedInternal();
    @Positive
    }

    @Positive
    private static class P11SecretKey extends P11Key implements SecretKey {

    @Positive
        public String getFormat();

    @Positive
        byte[] getEncodedInternal();
    @Positive
    }

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    private static class P11TlsMasterSecretKey extends P11SecretKey implements TlsMasterSecret {

    @Positive
        public int getMajorVersion();

    @Positive
        public int getMinorVersion();
    @Positive
    }

    @Positive
    private static final class P11RSAPrivateKey extends P11Key implements RSAPrivateCrtKey {

    @Positive
        public String getFormat();

    @Positive
        synchronized byte[] getEncodedInternal();

    @Positive
        public BigInteger getModulus();

    @Positive
        public BigInteger getPublicExponent();

    @Positive
        public BigInteger getPrivateExponent();

    @Positive
        public BigInteger getPrimeP();

    @Positive
        public BigInteger getPrimeQ();

    @Positive
        public BigInteger getPrimeExponentP();

    @Positive
        public BigInteger getPrimeExponentQ();

    @Positive
        public BigInteger getCrtCoefficient();
    @Positive
    }

    @Positive
    private static final class P11RSAPrivateNonCRTKey extends P11Key implements RSAPrivateKey {

    @Positive
        public String getFormat();

    @Positive
        synchronized byte[] getEncodedInternal();

    @Positive
        public BigInteger getModulus();

    @Positive
        public BigInteger getPrivateExponent();
    @Positive
    }

    @Positive
    private static final class P11RSAPublicKey extends P11Key implements RSAPublicKey {

    @Positive
        public String getFormat();

    @Positive
        synchronized byte[] getEncodedInternal();

    @Positive
        public BigInteger getModulus();

    @Positive
        public BigInteger getPublicExponent();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private static final class P11DSAPublicKey extends P11Key implements DSAPublicKey {

    @Positive
        public String getFormat();

    @Positive
        synchronized byte[] getEncodedInternal();

    @Positive
        public BigInteger getY();

    @Positive
        public DSAParams getParams();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private static final class P11DSAPrivateKey extends P11Key implements DSAPrivateKey {

    @Positive
        public String getFormat();

    @Positive
        synchronized byte[] getEncodedInternal();

    @Positive
        public BigInteger getX();

    @Positive
        public DSAParams getParams();
    @Positive
    }

    @Positive
    private static final class P11DHPrivateKey extends P11Key implements DHPrivateKey {

    @Positive
        public String getFormat();

    @Positive
        synchronized byte[] getEncodedInternal();

    @Positive
        public BigInteger getX();

    @Positive
        public DHParameterSpec getParams();

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object obj);
    @Positive
    }

    @Positive
    private static final class P11DHPublicKey extends P11Key implements DHPublicKey {

    @Positive
        public String getFormat();

    @Positive
        synchronized byte[] getEncodedInternal();

    @Positive
        public BigInteger getY();

    @Positive
        public DHParameterSpec getParams();

    @Positive
        public String toString();

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object obj);
    @Positive
    }

    @Positive
    private static final class P11ECPrivateKey extends P11Key implements ECPrivateKey {

    @Positive
        public String getFormat();

    @Positive
        synchronized byte[] getEncodedInternal();

    @Positive
        public BigInteger getS();

    @Positive
        public ECParameterSpec getParams();
    @Positive
    }

    @Positive
    private static final class P11ECPublicKey extends P11Key implements ECPublicKey {

    @Positive
        public String getFormat();

    @Positive
        synchronized byte[] getEncodedInternal();

    @Positive
        public ECPoint getW();

    @Positive
        public ECParameterSpec getParams();

    @Positive
        public String toString();
    @Positive
    }
    @Positive
}

    @Positive
final class NativeKeyHolder {

    @Positive
    static void decWrapperKeyRef();

    @Positive
    long getKeyID() throws ProviderException;

    @Positive
    void releaseKeyID();
    @Positive
}

    @Positive
final class SessionKeyRef extends PhantomReference<P11Key> {

    @Positive
    void registerNativeKey(long newKeyID, Session newSession);

    @Positive
    void removeNativeKey();

    @Positive
    void dispose();
    @Positive
}

// CFWR semantic augmentation - variant 1
