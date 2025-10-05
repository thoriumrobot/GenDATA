/*
 * Copyright (c) 2016, Oracle and/or its affiliates. All rights reserved.
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
package jdk.internal.jshell.tool;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.nio.file.AccessDeniedException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import static java.util.stream.Collectors.joining;
import static jdk.internal.jshell.tool.JShellTool.RECORD_SEPARATOR;
import static jdk.internal.jshell.tool.JShellTool.getResource;
import static jdk.internal.jshell.tool.JShellTool.readResource;
import static jdk.internal.jshell.tool.JShellTool.toPathResolvingUserHome;

class Startup {

    private static class StartupEntry {

        String storedForm();

        @Override
        public String toString();

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object o);
    }

    @Override
    public String toString();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    boolean isEmpty();

    boolean isDefault();

    String storedForm();

    String show(boolean isRetained);

    String showDetail();

    static Startup unpack(String storedForm, MessageHandler mh);

    static Startup fromFileList(List<String> fns, String context, MessageHandler mh);

    static Startup noStartup();

    static Startup defaultStartup(MessageHandler mh);
}
