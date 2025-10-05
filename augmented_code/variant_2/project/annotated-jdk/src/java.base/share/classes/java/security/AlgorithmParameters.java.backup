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
import java.io.*;
import java.security.spec.AlgorithmParameterSpec;
import java.security.spec.InvalidParameterSpecException;
import java.util.Objects;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class AlgorithmParameters {

    protected AlgorithmParameters(AlgorithmParametersSpi paramSpi, Provider provider, String algorithm) {
    }

    public final String getAlgorithm();

    public static AlgorithmParameters getInstance(String algorithm) throws NoSuchAlgorithmException;

    public static AlgorithmParameters getInstance(String algorithm, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    public static AlgorithmParameters getInstance(String algorithm, Provider provider) throws NoSuchAlgorithmException;

    public final Provider getProvider();

    public final void init(AlgorithmParameterSpec paramSpec) throws InvalidParameterSpecException;

    public final void init(byte[] params) throws IOException;

    public final void init(byte[] params, String format) throws IOException;

    public final <T extends AlgorithmParameterSpec> T getParameterSpec(Class<T> paramSpec) throws InvalidParameterSpecException;

    public final byte[] getEncoded() throws IOException;

    public final byte[] getEncoded(String format) throws IOException;

    public final String toString();
}
