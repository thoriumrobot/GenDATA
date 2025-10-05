/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.pkcs10;

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
import java.io.PrintStream;
    @Positive
import java.io.IOException;
    @Positive
import java.math.BigInteger;
    @Positive
import java.security.*;
    @Positive
import java.util.Base64;
    @Positive
import sun.security.util.*;
    @Positive
import sun.security.x509.AlgorithmId;
    @Positive
import sun.security.x509.X509Key;
    @Positive
import sun.security.x509.X500Name;
    @Positive
import sun.security.util.SignatureUtil;

    @Positive
public class PKCS10 {

    @Positive
    public PKCS10(PublicKey publicKey) {
    @Positive
    }

    @Positive
    public PKCS10(PublicKey publicKey, PKCS10Attributes attributes) {
    @Positive
    }

    @Positive
    public PKCS10(byte[] data) throws IOException, SignatureException, NoSuchAlgorithmException {
    @Positive
    }

    @Positive
    public void encodeAndSign(X500Name subject, PrivateKey key, String algorithm) throws IOException, SignatureException, NoSuchAlgorithmException, InvalidKeyException;

    @Positive
    public X500Name getSubjectName();

    @Positive
    public PublicKey getSubjectPublicKeyInfo();

    @Positive
    public String getSigAlg();

    @Positive
    public PKCS10Attributes getAttributes();

    @Positive
    public byte[] getEncoded();

    @Positive
    public void print(PrintStream out) throws IOException, SignatureException;

    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);

    @Positive
    public int hashCode();
    @Positive
}

// CFWR semantic augmentation - variant 0
