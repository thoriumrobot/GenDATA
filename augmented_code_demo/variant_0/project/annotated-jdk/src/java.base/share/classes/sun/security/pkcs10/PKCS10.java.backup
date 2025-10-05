/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.pkcs10;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.PrintStream;
import java.io.IOException;
import java.math.BigInteger;
import java.security.*;
import java.util.Base64;
import sun.security.util.*;
import sun.security.x509.AlgorithmId;
import sun.security.x509.X509Key;
import sun.security.x509.X500Name;
import sun.security.util.SignatureUtil;

public class PKCS10 {

    public PKCS10(PublicKey publicKey) {
    }

    public PKCS10(PublicKey publicKey, PKCS10Attributes attributes) {
    }

    public PKCS10(byte[] data) throws IOException, SignatureException, NoSuchAlgorithmException {
    }

    public void encodeAndSign(X500Name subject, PrivateKey key, String algorithm) throws IOException, SignatureException, NoSuchAlgorithmException, InvalidKeyException;

    public X500Name getSubjectName();

    public PublicKey getSubjectPublicKeyInfo();

    public String getSigAlg();

    public PKCS10Attributes getAttributes();

    public byte[] getEncoded();

    public void print(PrintStream out) throws IOException, SignatureException;

    public String toString();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    public int hashCode();
}
