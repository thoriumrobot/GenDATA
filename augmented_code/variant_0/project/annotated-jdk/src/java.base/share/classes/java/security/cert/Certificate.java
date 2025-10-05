/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.util.Arrays;
    @Positive
import java.security.Provider;
    @Positive
import java.security.PublicKey;
    @Positive
import java.security.NoSuchAlgorithmException;
    @Positive
import java.security.NoSuchProviderException;
    @Positive
import java.security.InvalidKeyException;
    @Positive
import java.security.SignatureException;
    @Positive
import sun.security.x509.X509CertImpl;

    @Positive
public abstract class Certificate implements java.io.Serializable {

    @Positive
    protected Certificate(String type) {
    @Positive
    }

    @Positive
    public final String getType();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);

    @Positive
    public int hashCode();

    @Positive
    public abstract byte[] getEncoded() throws CertificateEncodingException;

    @Positive
    public abstract void verify(PublicKey key) throws CertificateException, NoSuchAlgorithmException, InvalidKeyException, NoSuchProviderException, SignatureException;

    @Positive
    public abstract void verify(PublicKey key, String sigProvider) throws CertificateException, NoSuchAlgorithmException, InvalidKeyException, NoSuchProviderException, SignatureException;

    @Positive
    public void verify(PublicKey key, Provider sigProvider) throws CertificateException, NoSuchAlgorithmException, InvalidKeyException, SignatureException;

    @Positive
    public abstract String toString();

    @Positive
    public abstract PublicKey getPublicKey();

    @Positive
    protected static class CertificateRep implements java.io.Serializable {

    @Positive
        protected CertificateRep(String type, byte[] data) {
    @Positive
        }

    @Positive
        @java.io.Serial
    @Positive
        protected Object readResolve() throws java.io.ObjectStreamException;
    @Positive
    }

    @Positive
    @java.io.Serial
    @Positive
    protected Object writeReplace() throws java.io.ObjectStreamException;
    @Positive
}
