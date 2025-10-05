/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.InvalidAlgorithmParameterException;
    @Positive
import java.security.NoSuchAlgorithmException;
    @Positive
import java.security.NoSuchProviderException;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.Provider;
    @Positive
import java.security.Security;
    @Positive
import java.util.Objects;
    @Positive
import sun.security.jca.*;
    @Positive
import sun.security.jca.GetInstance.Instance;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class CertPathValidator {

    @Positive
    protected CertPathValidator(CertPathValidatorSpi validatorSpi, Provider provider, String algorithm) {
    @Positive
    }

    @Positive
    public static CertPathValidator getInstance(String algorithm) throws NoSuchAlgorithmException;

    @Positive
    public static CertPathValidator getInstance(String algorithm, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    @Positive
    public static CertPathValidator getInstance(String algorithm, Provider provider) throws NoSuchAlgorithmException;

    @Positive
    public final Provider getProvider();

    @Positive
    public final String getAlgorithm();

    @Positive
    public final CertPathValidatorResult validate(CertPath certPath, CertPathParameters params) throws CertPathValidatorException, InvalidAlgorithmParameterException;

    @Positive
    public static final String getDefaultType();

    @Positive
    public final CertPathChecker getRevocationChecker();
    @Positive
}

// CFWR semantic augmentation - variant 1
