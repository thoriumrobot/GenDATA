/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.pkcs11;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.*;
import java.lang.ref.*;
import java.math.BigInteger;
import java.util.*;
import java.security.*;
import java.security.interfaces.*;
import java.security.spec.*;
import javax.crypto.*;
import javax.crypto.interfaces.*;
import javax.crypto.spec.*;
import sun.security.rsa.RSAUtil.KeyType;
import sun.security.rsa.RSAPublicKeyImpl;
import sun.security.rsa.RSAPrivateCrtKeyImpl;
import sun.security.internal.interfaces.TlsMasterSecret;
import sun.security.pkcs11.wrapper.*;
import static sun.security.pkcs11.TemplateManager.O_GENERATE;
import static sun.security.pkcs11.wrapper.PKCS11Constants.*;
import sun.security.util.DerValue;
import sun.security.util.Length;
import sun.security.util.ECUtil;
import sun.security.jca.JCAUtil;

abstract class P11Key implements Key, Length {

    public long getKeyID();

    public void releaseKeyID();

    public final String getAlgorithm();

    public final byte[] getEncoded();

    static boolean drainRefQueue();

    abstract byte[] getEncodedInternal();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    protected Object writeReplace() throws ObjectStreamException;

    public String toString();

    @Override
    public int length();

    boolean isPublic();

    boolean isPrivate();

    boolean isSecret();

    void fetchAttributes(CK_ATTRIBUTE[] attributes);

    static SecretKey secretKey(Session session, long keyID, String algorithm, int keyLength, CK_ATTRIBUTE[] attributes);

    static SecretKey masterSecretKey(Session session, long keyID, String algorithm, int keyLength, CK_ATTRIBUTE[] attributes, int major, int minor);

    static PublicKey publicKey(Session session, long keyID, String algorithm, int keyLength, CK_ATTRIBUTE[] attributes);

    static PrivateKey privateKey(Session session, long keyID, String algorithm, int keyLength, CK_ATTRIBUTE[] attributes);

    private static final class P11PrivateKey extends P11Key implements PrivateKey {

        public String getFormat();

        byte[] getEncodedInternal();
    }

    private static class P11SecretKey extends P11Key implements SecretKey {

        public String getFormat();

        byte[] getEncodedInternal();
    }

    @SuppressWarnings("deprecation")
    private static class P11TlsMasterSecretKey extends P11SecretKey implements TlsMasterSecret {

        public int getMajorVersion();

        public int getMinorVersion();
    }

    private static final class P11RSAPrivateKey extends P11Key implements RSAPrivateCrtKey {

        public String getFormat();

        synchronized byte[] getEncodedInternal();

        public BigInteger getModulus();

        public BigInteger getPublicExponent();

        public BigInteger getPrivateExponent();

        public BigInteger getPrimeP();

        public BigInteger getPrimeQ();

        public BigInteger getPrimeExponentP();

        public BigInteger getPrimeExponentQ();

        public BigInteger getCrtCoefficient();
    }

    private static final class P11RSAPrivateNonCRTKey extends P11Key implements RSAPrivateKey {

        public String getFormat();

        synchronized byte[] getEncodedInternal();

        public BigInteger getModulus();

        public BigInteger getPrivateExponent();
    }

    private static final class P11RSAPublicKey extends P11Key implements RSAPublicKey {

        public String getFormat();

        synchronized byte[] getEncodedInternal();

        public BigInteger getModulus();

        public BigInteger getPublicExponent();

        public String toString();
    }

    private static final class P11DSAPublicKey extends P11Key implements DSAPublicKey {

        public String getFormat();

        synchronized byte[] getEncodedInternal();

        public BigInteger getY();

        public DSAParams getParams();

        public String toString();
    }

    private static final class P11DSAPrivateKey extends P11Key implements DSAPrivateKey {

        public String getFormat();

        synchronized byte[] getEncodedInternal();

        public BigInteger getX();

        public DSAParams getParams();
    }

    private static final class P11DHPrivateKey extends P11Key implements DHPrivateKey {

        public String getFormat();

        synchronized byte[] getEncodedInternal();

        public BigInteger getX();

        public DHParameterSpec getParams();

        public int hashCode();

        public boolean equals(Object obj);
    }

    private static final class P11DHPublicKey extends P11Key implements DHPublicKey {

        public String getFormat();

        synchronized byte[] getEncodedInternal();

        public BigInteger getY();

        public DHParameterSpec getParams();

        public String toString();

        public int hashCode();

        public boolean equals(Object obj);
    }

    private static final class P11ECPrivateKey extends P11Key implements ECPrivateKey {

        public String getFormat();

        synchronized byte[] getEncodedInternal();

        public BigInteger getS();

        public ECParameterSpec getParams();
    }

    private static final class P11ECPublicKey extends P11Key implements ECPublicKey {

        public String getFormat();

        synchronized byte[] getEncodedInternal();

        public ECPoint getW();

        public ECParameterSpec getParams();

        public String toString();
    }
}

final class NativeKeyHolder {

    static void decWrapperKeyRef();

    long getKeyID() throws ProviderException;

    void releaseKeyID();
}

final class SessionKeyRef extends PhantomReference<P11Key> {

    void registerNativeKey(long newKeyID, Session newSession);

    void removeNativeKey();

    void dispose();
}
