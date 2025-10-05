/*
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.guieffect.qual.SafeEffect;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.desktop.AboutEvent;
import java.awt.desktop.AboutHandler;
import java.awt.desktop.OpenFilesHandler;
import java.awt.desktop.OpenURIEvent;
import java.awt.desktop.OpenURIHandler;
import java.awt.desktop.PreferencesEvent;
import java.awt.desktop.PreferencesHandler;
import java.awt.desktop.PrintFilesHandler;
import java.awt.desktop.QuitHandler;
import java.awt.desktop.QuitStrategy;
import java.awt.desktop.SystemEventListener;
import java.awt.peer.DesktopPeer;
import java.io.File;
import java.io.FilePermission;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Objects;
import javax.swing.JMenuBar;
import sun.awt.SunToolkit;
import sun.security.util.SecurityConstants;

@AnnotatedFor({ "guieffect", "interning" })
@UsesObjectEquals
public class Desktop {

    public static enum Action {

        OPEN,
        EDIT,
        PRINT,
        MAIL,
        BROWSE,
        APP_EVENT_FOREGROUND,
        APP_EVENT_HIDDEN,
        APP_EVENT_REOPENED,
        APP_EVENT_SCREEN_SLEEP,
        APP_EVENT_SYSTEM_SLEEP,
        APP_EVENT_USER_SESSION,
        APP_ABOUT,
        APP_PREFERENCES,
        APP_OPEN_FILE,
        APP_PRINT_FILE,
        APP_OPEN_URI,
        APP_QUIT_HANDLER,
        APP_QUIT_STRATEGY,
        APP_SUDDEN_TERMINATION,
        APP_REQUEST_FOREGROUND,
        APP_HELP_VIEWER,
        APP_MENU_BAR,
        BROWSE_FILE_DIR,
        MOVE_TO_TRASH
    }

    @SafeEffect
    public static synchronized Desktop getDesktop();

    @SafeEffect
    public static boolean isDesktopSupported();

    @SafeEffect
    public boolean isSupported(Action action);

    @SafeEffect
    public void open(File file) throws IOException;

    @SafeEffect
    public void edit(File file) throws IOException;

    @SafeEffect
    public void print(File file) throws IOException;

    @SafeEffect
    public void browse(URI uri) throws IOException;

    @SafeEffect
    public void mail() throws IOException;

    @SafeEffect
    public void mail(URI mailtoURI) throws IOException;

    public void addAppEventListener(final SystemEventListener listener);

    public void removeAppEventListener(final SystemEventListener listener);

    public void setAboutHandler(final AboutHandler aboutHandler);

    public void setPreferencesHandler(final PreferencesHandler preferencesHandler);

    public void setOpenFileHandler(final OpenFilesHandler openFileHandler);

    public void setPrintFileHandler(final PrintFilesHandler printFileHandler);

    public void setOpenURIHandler(final OpenURIHandler openURIHandler);

    public void setQuitHandler(final QuitHandler quitHandler);

    public void setQuitStrategy(final QuitStrategy strategy);

    public void enableSuddenTermination();

    public void disableSuddenTermination();

    public void requestForeground(final boolean allWindows);

    public void openHelpViewer();

    public void setDefaultMenuBar(final JMenuBar menuBar);

    public void browseFileDirectory(File file);

    @SuppressWarnings("removal")
    public boolean moveToTrash(File file);
}
