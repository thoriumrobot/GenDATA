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

import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import java.io.*;
import java.util.*;
import static java.util.Locale.ENGLISH;
import java.lang.ref.*;
import java.lang.reflect.*;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.concurrent.ConcurrentHashMap;

public abstract class Provider extends Properties {

    @Deprecated()
    protected Provider(String name, double version, String info) {
    }

    protected Provider(String name, String versionStr, String info) {
    }

    public Provider configure(String configArg);

    public boolean isConfigured();

    public String getName();

    @Deprecated()
    public double getVersion();

    public String getVersionStr();

    public String getInfo();

    public String toString();

    @Override
    public synchronized void clear();

    @Override
    public synchronized void load(InputStream inStream) throws IOException;

    @Override
    public synchronized void putAll(Map<?, ?> t);

    @Override
    public synchronized Set<Map.Entry<Object, Object>> entrySet();

    @Override
    public Set<Object> keySet();

    @Override
    public Collection<Object> values();

    @Override
    public synchronized Object put(Object key, Object value);

    @Override
    public synchronized Object putIfAbsent(Object key, Object value);

    @Override
    public synchronized Object remove(Object key);

    @Override
    public synchronized boolean remove(@UnknownSignedness Object key, @UnknownSignedness Object value);

    @Override
    public synchronized boolean replace(Object key, Object oldValue, Object newValue);

    @Override
    public synchronized Object replace(Object key, Object value);

    @Override
    public synchronized void replaceAll(BiFunction<? super Object, ? super Object, ? extends Object> function);

    @Override
    @PolyNull
    public synchronized Object compute(Object key, BiFunction<? super Object, ? super Object, ? extends @PolyNull Object> remappingFunction);

    @Override
    @PolyNull
    public synchronized Object computeIfAbsent(Object key, Function<? super Object, ? extends @PolyNull Object> mappingFunction);

    @Override
    @PolyNull
    public synchronized Object computeIfPresent(Object key, BiFunction<? super Object, ? super Object, ? extends @PolyNull Object> remappingFunction);

    @Override
    @PolyNull
    public synchronized Object merge(Object key, Object value, BiFunction<? super Object, ? super Object, ? extends @PolyNull Object> remappingFunction);

    @Override
    public Object get(Object key);

    @Override
    public synchronized Object getOrDefault(Object key, Object defaultValue);

    @Override
    public synchronized void forEach(BiConsumer<? super Object, ? super Object> action);

    @Override
    public Enumeration<Object> keys();

    @Override
    public Enumeration<Object> elements();

    public String getProperty(String key);

    private static class ServiceKey {

        public int hashCode();

        public boolean equals(Object obj);

        boolean matches(String type, String algorithm);
    }

    public Service getService(String type, String algorithm);

    public synchronized Set<Service> getServices();

    protected void putService(Service s);

    synchronized Service getDefaultSecureRandomService();

    protected void removeService(Service s);

    private static class UString {

        public int hashCode();

        public boolean equals(Object obj);

        public String toString();
    }

    private static class EngineDescription {

        Class<?> getConstructorParameterClass() throws ClassNotFoundException;
    }

    public static class Service {

        void addAttribute(String type, String value);

        public Service(Provider provider, String type, String algorithm, String className, List<String> aliases, Map<String, String> attributes) {
        }

        public final String getType();

        public final String getAlgorithm();

        public final Provider getProvider();

        public final String getClassName();

        public final String getAttribute(String name);

        public Object newInstance(Object constructorParameter) throws NoSuchAlgorithmException;

        public boolean supportsParameter(Object parameter);

        public String toString();
    }
}
