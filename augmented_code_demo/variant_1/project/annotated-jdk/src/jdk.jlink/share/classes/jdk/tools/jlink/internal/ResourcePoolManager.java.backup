/*
 * Copyright (c) 2015, 2017, Oracle and/or its affiliates. All rights reserved.
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
package jdk.tools.jlink.internal;

import org.checkerframework.dataflow.qual.Pure;
import java.lang.module.ModuleDescriptor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import jdk.internal.jimage.decompressor.CompressedResourceHeader;
import jdk.internal.module.Resources;
import jdk.internal.module.ModuleInfo;
import jdk.internal.module.ModuleInfo.Attributes;
import jdk.internal.module.ModuleTarget;
import jdk.tools.jlink.plugin.ResourcePool;
import jdk.tools.jlink.plugin.ResourcePoolBuilder;
import jdk.tools.jlink.plugin.ResourcePoolEntry;
import jdk.tools.jlink.plugin.ResourcePoolModule;
import jdk.tools.jlink.plugin.ResourcePoolModuleView;
import jdk.tools.jlink.plugin.PluginException;

public class ResourcePoolManager {

    static Attributes readModuleAttributes(ResourcePoolModule mod);

    public static boolean isNamedPackageResource(String path);

    class ResourcePoolModuleImpl implements ResourcePoolModule {

        @Override
        public String name();

        @Override
        public Optional<ResourcePoolEntry> findEntry(String path);

        @Override
        public ModuleDescriptor descriptor();

        @Override
        public String targetPlatform();

        @Override
        public Set<String> packages();

        @Override
        public String toString();

        @Override
        public Stream<ResourcePoolEntry> entries();

        @Override
        public int entryCount();
    }

    public class ResourcePoolImpl implements ResourcePool {

        @Override
        public ResourcePoolModuleView moduleView();

        @Override
        public Stream<ResourcePoolEntry> entries();

        @Override
        public int entryCount();

        @Override
        public Optional<ResourcePoolEntry> findEntry(String path);

        @Override
        public Optional<ResourcePoolEntry> findEntryInContext(String path, ResourcePoolEntry context);

        @Override
        @Pure
        public boolean contains(ResourcePoolEntry data);

        @Override
        public boolean isEmpty();

        @Override
        public ByteOrder byteOrder();

        public StringTable getStringTable();
    }

    class ResourcePoolBuilderImpl implements ResourcePoolBuilder {

        @Override
        public void add(ResourcePoolEntry data);

        @Override
        public ResourcePool build();
    }

    class ResourcePoolModuleViewImpl implements ResourcePoolModuleView {

        @Override
        public Optional<ResourcePoolModule> findModule(String name);

        @Override
        public Stream<ResourcePoolModule> modules();

        @Override
        public int moduleCount();
    }

    public ResourcePoolManager() {
    }

    public ResourcePoolManager(ByteOrder order) {
    }

    public ResourcePoolManager(ByteOrder order, StringTable table) {
    }

    public ResourcePool resourcePool();

    public ResourcePoolBuilder resourcePoolBuilder();

    public ResourcePoolModuleView moduleView();

    public void add(ResourcePoolEntry data);

    public Optional<ResourcePoolModule> findModule(String name);

    public Stream<ResourcePoolModule> modules();

    public int moduleCount();

    public Stream<ResourcePoolEntry> entries();

    public int entryCount();

    public Optional<ResourcePoolEntry> findEntry(String path);

    public Optional<ResourcePoolEntry> findEntryInContext(String path, ResourcePoolEntry context);

    @Pure
    public boolean contains(ResourcePoolEntry data);

    public boolean isEmpty();

    public ByteOrder byteOrder();

    public StringTable getStringTable();

    public static final class CompressedModuleData extends ByteArrayResourcePoolEntry {

        public long getUncompressedSize();

        @Override
        public boolean equals(Object other);

        @Override
        public int hashCode();
    }

    public static CompressedModuleData newCompressedResource(ResourcePoolEntry original, ByteBuffer compressed, String plugin, String pluginConfig, StringTable strings, ByteOrder order);
}
