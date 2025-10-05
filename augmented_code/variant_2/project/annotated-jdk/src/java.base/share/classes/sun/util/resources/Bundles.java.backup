/*
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.util.resources;

import org.checkerframework.checker.nullness.qual.Nullable;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.SoftReference;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.MissingResourceException;
import java.util.Objects;
import java.util.ResourceBundle;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.spi.ResourceBundleProvider;
import jdk.internal.access.JavaUtilResourceBundleAccess;
import jdk.internal.access.SharedSecrets;

public abstract class Bundles {

    public static ResourceBundle of(String baseName, Locale locale, Strategy strategy);

    public static String toOtherBundleName(String baseName, String bundleName, Locale locale);

    public interface Strategy {

        List<Locale> getCandidateLocales(String baseName, Locale locale);

        String toBundleName(String baseName, Locale locale);

        Class<? extends ResourceBundleProvider> getResourceBundleProviderType(String baseName, Locale locale);
    }

    private static interface CacheKeyReference {

        CacheKey getCacheKey();
    }

    private static class BundleReference extends SoftReference<ResourceBundle> implements CacheKeyReference {

        @Override
        public CacheKey getCacheKey();
    }

    private static class CacheKey implements Cloneable {

        String getName();

        CacheKey setName(String baseName);

        Locale getLocale();

        CacheKey setLocale(Locale locale);

        ServiceLoader<ResourceBundleProvider> getProviders();

        void setProviders(ServiceLoader<ResourceBundleProvider> providers);

        @Override
        public boolean equals(Object other);

        @Override
        public int hashCode();

        @Override
        public Object clone();

        @Override
        public String toString();
    }
}
