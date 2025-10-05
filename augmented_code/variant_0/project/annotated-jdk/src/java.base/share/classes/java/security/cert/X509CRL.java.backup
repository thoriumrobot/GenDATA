/*
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.security.cert;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.security.*;
import java.security.spec.*;
import javax.security.auth.x500.X500Principal;
import java.math.BigInteger;
import java.util.Date;
import java.util.Set;
import java.util.Arrays;
import sun.security.x509.X509CRLImpl;
import sun.security.util.SignatureUtil;

public abstract class X509CRL extends CRL implements X509Extension {

    protected X509CRL() {
    }

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    public int hashCode();

    public abstract byte[] getEncoded() throws CRLException;

    public abstract void verify(PublicKey key) throws CRLException, NoSuchAlgorithmException, InvalidKeyException, NoSuchProviderException, SignatureException;

    public abstract void verify(PublicKey key, String sigProvider) throws CRLException, NoSuchAlgorithmException, InvalidKeyException, NoSuchProviderException, SignatureException;

    public void verify(PublicKey key, Provider sigProvider) throws CRLException, NoSuchAlgorithmException, InvalidKeyException, SignatureException;

    public abstract int getVersion();

    @Deprecated()
    public abstract Principal getIssuerDN();

    public X500Principal getIssuerX500Principal();

    public abstract Date getThisUpdate();

    public abstract Date getNextUpdate();

    public abstract X509CRLEntry getRevokedCertificate(BigInteger serialNumber);

    public X509CRLEntry getRevokedCertificate(X509Certificate certificate);

    public abstract Set<? extends X509CRLEntry> getRevokedCertificates();

    public abstract byte[] getTBSCertList() throws CRLException;

    public abstract byte[] getSignature();

    public abstract String getSigAlgName();

    public abstract String getSigAlgOID();

    public abstract byte[] getSigAlgParams();
}
