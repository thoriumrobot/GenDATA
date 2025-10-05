/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.awt.shell;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.image.AbstractMultiResolutionImage;
import java.awt.image.BufferedImage;
import java.awt.image.ImageObserver;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serial;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import javax.swing.SwingConstants;

@SuppressWarnings("serial")
final class Win32ShellFolder2 extends ShellFolder {

    public static final int DESKTOP;

    public static final int INTERNET;

    public static final int PROGRAMS;

    public static final int CONTROLS;

    public static final int PRINTERS;

    public static final int PERSONAL;

    public static final int FAVORITES;

    public static final int STARTUP;

    public static final int RECENT;

    public static final int SENDTO;

    public static final int BITBUCKET;

    public static final int STARTMENU;

    public static final int DESKTOPDIRECTORY;

    public static final int DRIVES;

    public static final int NETWORK;

    public static final int NETHOOD;

    public static final int FONTS;

    public static final int TEMPLATES;

    public static final int COMMON_STARTMENU;

    public static final int COMMON_PROGRAMS;

    public static final int COMMON_STARTUP;

    public static final int COMMON_DESKTOPDIRECTORY;

    public static final int APPDATA;

    public static final int PRINTHOOD;

    public static final int ALTSTARTUP;

    public static final int COMMON_ALTSTARTUP;

    public static final int COMMON_FAVORITES;

    public static final int INTERNET_CACHE;

    public static final int COOKIES;

    public static final int HISTORY;

    public static final int ATTRIB_CANCOPY;

    public static final int ATTRIB_CANMOVE;

    public static final int ATTRIB_CANLINK;

    public static final int ATTRIB_CANRENAME;

    public static final int ATTRIB_CANDELETE;

    public static final int ATTRIB_HASPROPSHEET;

    public static final int ATTRIB_DROPTARGET;

    public static final int ATTRIB_LINK;

    public static final int ATTRIB_SHARE;

    public static final int ATTRIB_READONLY;

    public static final int ATTRIB_GHOSTED;

    public static final int ATTRIB_HIDDEN;

    public static final int ATTRIB_FILESYSANCESTOR;

    public static final int ATTRIB_FOLDER;

    public static final int ATTRIB_FILESYSTEM;

    public static final int ATTRIB_HASSUBFOLDER;

    public static final int ATTRIB_VALIDATE;

    public static final int ATTRIB_REMOVABLE;

    public static final int ATTRIB_COMPRESSED;

    public static final int ATTRIB_BROWSABLE;

    public static final int ATTRIB_NONENUMERATED;

    public static final int ATTRIB_NEWCONTENT;

    public static final int SHGDN_NORMAL;

    public static final int SHGDN_INFOLDER;

    public static final int SHGDN_INCLUDE_NONFILESYS;

    public static final int SHGDN_FORADDRESSBAR;

    public static final int SHGDN_FORPARSING;

    public enum SystemIcon {

        IDI_APPLICATION(32512),
        IDI_HAND(32513),
        IDI_ERROR(32513),
        IDI_QUESTION(32514),
        IDI_EXCLAMATION(32515),
        IDI_WARNING(32515),
        IDI_ASTERISK(32516),
        IDI_INFORMATION(32516),
        IDI_WINLOGO(32517);

        public int getIconID();
    }

    static final class KnownFolderDefinition {
    }

    static final class KnownLibraries {
    }

    static class FolderDisposer implements sun.java2d.DisposerRecord {

        public void dispose();
    }

    static Win32ShellFolder2 createShellFolder(Win32ShellFolder2 parent, long pIDL) throws InterruptedException;

    public void setIsPersonal();

    @Serial
    protected Object writeReplace() throws java.io.ObjectStreamException;

    protected void dispose();

    static native long getNextPIDLEntry(long pIDL);

    static native long copyFirstPIDLEntry(long pIDL);

    static native void releasePIDL(long pIDL);

    public long getParentIShellFolder();

    public long getRelativePIDL();

    public Win32ShellFolder2 getDesktop();

    public long getDesktopIShellFolder();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public boolean isFileSystem();

    public boolean hasAttribute(final int attribute);

    static String getFileSystemPath(final int csidl) throws IOException, InterruptedException;

    public File getParentFile();

    public boolean isDirectory();

    public File[] listFiles(final boolean includeHiddenFiles);

    Win32ShellFolder2 getChildByPath(final String filePath) throws InterruptedException;

    public boolean isLink();

    public boolean isHidden();

    public ShellFolder getLinkLocation();

    long parseDisplayName(final String name) throws IOException, InterruptedException;

    public String getDisplayName();

    public String getFolderType();

    public String getExecutableType();

    static native int[] getStandardViewButton0(int iconIndex, boolean small);

    public Image getIcon(final boolean getLargeIcon);

    public Image getIcon(int width, int height);

    static Image getSystemIcon(SystemIcon iconType);

    static Image getShell32Icon(int iconID, int size);

    public File getCanonicalFile() throws IOException;

    public boolean isSpecial();

    public int compareTo(File file2);

    public ShellFolderColumnInfo[] getFolderColumns();

    public Object getFolderColumnValue(final int column);

    boolean isLibrary();

    public void sortChildren(final List<? extends File> files);

    private static class ColumnComparator implements Comparator<File> {

        public ColumnComparator(Win32ShellFolder2 shellFolder, int columnIdx) {
        }

        public int compare(final File o, final File o1);
    }

    static class MultiResolutionIconImage extends AbstractMultiResolutionImage {

        public MultiResolutionIconImage(int baseSize, Map<Integer, Image> resolutionVariants) {
        }

        public MultiResolutionIconImage(int baseSize, Image image) {
        }

        @Override
        public int getWidth(ImageObserver observer);

        @Override
        public int getHeight(ImageObserver observer);

        @Override
        protected Image getBaseImage();

        @Override
        public Image getResolutionVariant(double width, double height);

        @Override
        public List<Image> getResolutionVariants();
    }
}
