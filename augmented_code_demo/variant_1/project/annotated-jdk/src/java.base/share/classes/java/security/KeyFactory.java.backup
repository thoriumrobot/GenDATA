/*
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.security;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.*;
import java.security.Provider.Service;
import java.security.spec.KeySpec;
import java.security.spec.InvalidKeySpecException;
import sun.security.util.Debug;
import sun.security.jca.*;
import sun.security.jca.GetInstance.Instance;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class KeyFactory {

    protected KeyFactory(KeyFactorySpi keyFacSpi, Provider provider, String algorithm) {
    }

    public static KeyFactory getInstance(String algorithm) throws NoSuchAlgorithmException;

    public static KeyFactory getInstance(String algorithm, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    public static KeyFactory getInstance(String algorithm, Provider provider) throws NoSuchAlgorithmException;

    public final Provider getProvider();

    public final String getAlgorithm();

    public final PublicKey generatePublic(KeySpec keySpec) throws InvalidKeySpecException;

    public final PrivateKey generatePrivate(KeySpec keySpec) throws InvalidKeySpecException;

    public final <T extends KeySpec> T getKeySpec(Key key, Class<T> keySpec) throws InvalidKeySpecException;

    public final Key translateKey(Key key) throws InvalidKeyException;
}
