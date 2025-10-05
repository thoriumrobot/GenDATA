/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.util.concurrent.ConcurrentHashMap;
import java.io.*;
import java.net.URL;
import jdk.internal.event.EventHelper;
import jdk.internal.event.SecurityPropertyModificationEvent;
import jdk.internal.access.SharedSecrets;
import jdk.internal.util.StaticProperty;
import sun.security.util.Debug;
import sun.security.util.PropertyExpander;
import sun.security.jca.*;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public final class Security {

    private static class ProviderProperty {
    }

    @Deprecated
    public static String getAlgorithmProperty(String algName, String propName);

    public static synchronized int insertProviderAt(Provider provider, int position);

    public static int addProvider(Provider provider);

    public static synchronized void removeProvider(String name);

    public static Provider[] getProviders();

    public static Provider getProvider(String name);

    public static Provider[] getProviders(String filter);

    public static Provider[] getProviders(Map<String, String> filter);

    static Object[] getImpl(String algorithm, String type, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    static Object[] getImpl(String algorithm, String type, String provider, Object params) throws NoSuchAlgorithmException, NoSuchProviderException, InvalidAlgorithmParameterException;

    static Object[] getImpl(String algorithm, String type, Provider provider) throws NoSuchAlgorithmException;

    static Object[] getImpl(String algorithm, String type, Provider provider, Object params) throws NoSuchAlgorithmException, InvalidAlgorithmParameterException;

    public static String getProperty(String key);

    public static void setProperty(String key, String datum);

    static String[] getFilterComponents(String filterKey, String filterValue);

    public static Set<String> getAlgorithms(String serviceName);
}
