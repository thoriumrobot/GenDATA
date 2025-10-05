/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.security;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.*;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.io.*;
    @Positive
import java.net.URL;
    @Positive
import jdk.internal.event.EventHelper;
    @Positive
import jdk.internal.event.SecurityPropertyModificationEvent;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.util.StaticProperty;
    @Positive
import sun.security.util.Debug;
    @Positive
import sun.security.util.PropertyExpander;
    @Positive
import sun.security.jca.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Security {

    @Positive
    private static class ProviderProperty {
    @Positive
    }

    @Positive
    @Deprecated
    @Positive
    public static String getAlgorithmProperty(String algName, String propName);

    @Positive
    public static synchronized int insertProviderAt(Provider provider, int position);

    @Positive
    public static int addProvider(Provider provider);

    @Positive
    public static synchronized void removeProvider(String name);

    @Positive
    public static Provider[] getProviders();

    @Positive
    public static Provider getProvider(String name);

    @Positive
    public static Provider[] getProviders(String filter);

    @Positive
    public static Provider[] getProviders(Map<String, String> filter);

    @Positive
    static Object[] getImpl(String algorithm, String type, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    @Positive
    static Object[] getImpl(String algorithm, String type, String provider, Object params) throws NoSuchAlgorithmException, NoSuchProviderException, InvalidAlgorithmParameterException;

    @Positive
    static Object[] getImpl(String algorithm, String type, Provider provider) throws NoSuchAlgorithmException;

    @Positive
    static Object[] getImpl(String algorithm, String type, Provider provider, Object params) throws NoSuchAlgorithmException, InvalidAlgorithmParameterException;

    @Positive
    public static String getProperty(String key);

    @Positive
    public static void setProperty(String key, String datum);

    @Positive
    static String[] getFilterComponents(String filterKey, String filterValue);

    @Positive
    public static Set<String> getAlgorithms(String serviceName);
    @Positive
}

// CFWR semantic augmentation - variant 1
