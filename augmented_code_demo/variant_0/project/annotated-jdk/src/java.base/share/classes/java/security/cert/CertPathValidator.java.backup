/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.security.AccessController;
import java.security.InvalidAlgorithmParameterException;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.PrivilegedAction;
import java.security.Provider;
import java.security.Security;
import java.util.Objects;
import sun.security.jca.*;
import sun.security.jca.GetInstance.Instance;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class CertPathValidator {

    protected CertPathValidator(CertPathValidatorSpi validatorSpi, Provider provider, String algorithm) {
    }

    public static CertPathValidator getInstance(String algorithm) throws NoSuchAlgorithmException;

    public static CertPathValidator getInstance(String algorithm, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    public static CertPathValidator getInstance(String algorithm, Provider provider) throws NoSuchAlgorithmException;

    public final Provider getProvider();

    public final String getAlgorithm();

    public final CertPathValidatorResult validate(CertPath certPath, CertPathParameters params) throws CertPathValidatorException, InvalidAlgorithmParameterException;

    public static final String getDefaultType();

    public final CertPathChecker getRevocationChecker();
}
