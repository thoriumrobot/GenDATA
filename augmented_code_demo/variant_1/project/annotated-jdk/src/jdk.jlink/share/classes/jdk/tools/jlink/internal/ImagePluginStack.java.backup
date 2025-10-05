/*
 * Copyright (c) 2015, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.module.ModuleDescriptor;
import java.nio.ByteOrder;
import java.util.*;
import java.util.stream.Stream;
import jdk.internal.jimage.decompressor.Decompressor;
import jdk.internal.module.ModuleInfo.Attributes;
import jdk.internal.module.ModuleTarget;
import jdk.tools.jlink.builder.ImageBuilder;
import jdk.tools.jlink.plugin.Plugin;
import jdk.tools.jlink.plugin.PluginException;
import jdk.tools.jlink.plugin.ResourcePool;
import jdk.tools.jlink.plugin.ResourcePoolEntry;
import jdk.tools.jlink.plugin.ResourcePoolModule;

public final class ImagePluginStack {

    public interface ImageProvider {

        ExecutableImage retrieve(ImagePluginStack stack) throws IOException;
    }

    public static final class OrderedResourcePoolManager extends ResourcePoolManager {

        class OrderedResourcePool extends ResourcePoolImpl {

            List<ResourcePoolEntry> getOrderedList();
        }

        public OrderedResourcePoolManager(ByteOrder order, StringTable table) {
        }

        @Override
        public ResourcePool resourcePool();

        @Override
        public void add(ResourcePoolEntry resource);

        List<ResourcePoolEntry> getOrderedList();
    }

    private final static class CheckOrderResourcePoolManager extends ResourcePoolManager {

        public CheckOrderResourcePoolManager(ByteOrder order, List<ResourcePoolEntry> orderedList, StringTable table) {
        }

        @Override
        public void add(ResourcePoolEntry resource);
    }

    private static final class PreVisitStrings implements StringTable {

        @Override
        public int addString(String str);

        @Override
        public String getString(int id);
    }

    public ImagePluginStack() {
    }

    public ImagePluginStack(ImageBuilder imageBuilder, List<Plugin> plugins, Plugin lastSorter) {
    }

    public ImagePluginStack(ImageBuilder imageBuilder, List<Plugin> plugins, Plugin lastSorter, boolean validate) {
    }

    public void operate(ImageProvider provider) throws Exception;

    public DataOutputStream getJImageFileOutputStream() throws IOException;

    public ImageBuilder getImageBuilder();

    public ResourcePool visitResources(ResourcePoolManager resources) throws Exception;

    private class LastPoolManager extends ResourcePoolManager {

        private class LastModule implements ResourcePoolModule {

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

        @Override
        public void add(ResourcePoolEntry resource);

        @Override
        public Optional<ResourcePoolModule> findModule(String name);

        @Override
        public Stream<ResourcePoolModule> modules();

        @Override
        public int moduleCount();

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
        public boolean contains(ResourcePoolEntry res);

        @Override
        public boolean isEmpty();

        @Override
        public ByteOrder byteOrder();
    }

    public void storeFiles(ResourcePool original, ResourcePool transformed, BasicImageWriter writer) throws Exception;

    public ExecutableImage getExecutableImage() throws IOException;
}
