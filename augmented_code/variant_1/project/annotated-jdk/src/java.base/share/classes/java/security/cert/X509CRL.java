/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package java.security.cert;

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
import java.security.*;
    @Positive
import java.security.spec.*;
    @Positive
import javax.security.auth.x500.X500Principal;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.Date;
    @Positive
import java.util.Set;
    @Positive
import java.util.Arrays;
    @Positive
import sun.security.x509.X509CRLImpl;
    @Positive
import sun.security.util.SignatureUtil;

    @Positive
public abstract class X509CRL extends CRL implements X509Extension {

    @Positive
    protected X509CRL() {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);

    @Positive
    public int hashCode();

    @Positive
    public abstract byte[] getEncoded() throws CRLException;

    @Positive
    public abstract void verify(PublicKey key) throws CRLException, NoSuchAlgorithmException, InvalidKeyException, NoSuchProviderException, SignatureException;

    @Positive
    public abstract void verify(PublicKey key, String sigProvider) throws CRLException, NoSuchAlgorithmException, InvalidKeyException, NoSuchProviderException, SignatureException;

    @Positive
    public void verify(PublicKey key, Provider sigProvider) throws CRLException, NoSuchAlgorithmException, InvalidKeyException, SignatureException;

    @Positive
    public abstract int getVersion();

    @Positive
    @Deprecated()
    @Positive
    public abstract Principal getIssuerDN();

    @Positive
    public X500Principal getIssuerX500Principal();

    @Positive
    public abstract Date getThisUpdate();

    @Positive
    public abstract Date getNextUpdate();

    @Positive
    public abstract X509CRLEntry getRevokedCertificate(BigInteger serialNumber);

    @Positive
    public X509CRLEntry getRevokedCertificate(X509Certificate certificate);

    @Positive
    public abstract Set<? extends X509CRLEntry> getRevokedCertificates();

    @Positive
    public abstract byte[] getTBSCertList() throws CRLException;

    @Positive
    public abstract byte[] getSignature();

    @Positive
    public abstract String getSigAlgName();

    @Positive
    public abstract String getSigAlgOID();

    @Positive
    public abstract byte[] getSigAlgParams();
    @Positive
}
