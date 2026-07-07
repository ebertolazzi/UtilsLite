# frozen_string_literal: true

require 'etc'
require 'fileutils'
require 'rake/clean'

begin
  require 'colorize'
rescue LoadError
  warn 'Install the colorize gem: gem install colorize'
  exit 1
end

# Avoid removing files named "core" from included libraries such as Eigen.
CLEAN.clear_exclude.exclude { |fn| fn.pathmap('%f').downcase == 'core' }

# Optional shared configuration, searched in parent directories.
config_files = [
  File.expand_path('../Rakefile_configure.rb', __dir__),
  File.expand_path('../../Rakefile_configure.rb', __dir__)
]

if (config = config_files.find { |path| File.exist?(path) })
  require config
else
  COMPILE_DEBUG      = false unless defined?(COMPILE_DEBUG)
  COMPILE_DYNAMIC    = false unless defined?(COMPILE_DYNAMIC)
  COMPILE_EXECUTABLE = true  unless defined?(COMPILE_EXECUTABLE)
end

OS = case RUBY_PLATFORM
     when /darwin/       then :mac
     when /linux|cygwin/ then :linux
     when /msys/         then :mingw
     else                     :win
     end

BUILD_TYPE = COMPILE_DEBUG ? 'Debug' : 'Release'

BUILD_OPTIONS = [
  "-DCMAKE_BUILD_TYPE=#{BUILD_TYPE}",
  "-DUTILS_ENABLE_TESTS=#{COMPILE_EXECUTABLE ? 'ON' : 'OFF'}",
  "-DUTILS_BUILD_SHARED=#{COMPILE_DYNAMIC ? 'ON' : 'OFF'}"
].join(' ')

PARALLEL = if OS == :win
             ''
           else
             "--parallel #{Etc.nprocessors}"
           end

def in_dir(path)
  FileUtils.mkdir_p(path)
  Dir.chdir(path) { yield }
end

def visual_studio_arch
  cl = `where cl.exe 2>NUL`.lines.first.to_s.strip

  case cl
  when /(x64|amd64)\\cl\.exe/i then 'x64'
  when /(bin|x86|amd32)\\cl\.exe/i then 'x86'
  else
    raise 'Cannot determine Visual Studio architecture. Run from a Visual Studio Developer Prompt.'
  end
end

def configure_and_build(bits: nil)
  FileUtils.rm_rf('lib')
  FileUtils.rm_rf('build')

  in_dir('build') do
    bits_opt = bits ? "-DBITS=#{bits}" : ''
    sh "cmake -G Ninja #{bits_opt} #{BUILD_OPTIONS} .."
    sh "cmake --build . --config #{BUILD_TYPE} --target install #{PARALLEL}"
  end
end

desc 'Default task: build'
task default: :build

desc 'Build with CMake/Ninja'
task :build do
  puts "Build (#{OS})".green

  bits = OS == :win ? visual_studio_arch : nil
  configure_and_build(bits: bits)
end

desc 'Run CTest from build/'
task :test do
  Dir.chdir('build') { sh 'ctest --output-on-failure' }
end

desc 'Run executables from bin/'
task :run do
  exes = if OS == :win || OS == :mingw
           Dir.glob('bin/*.exe')
         else
           Dir.glob('bin/*').select { |path| File.file?(path) && File.executable?(path) }
         end

  raise 'No executables found in bin/' if exes.empty?

  exes.sort.each do |exe|
    puts "execute #{exe}".yellow
    sh exe
  end
end

desc 'Clean build artifacts'
task :clean do
  FileUtils.rm_rf(%w[build lib])
end

desc 'Hard reset repository and submodules'
task :git_submodules do
  sh 'git reset --hard'
  sh 'git submodule sync --recursive'
  sh 'git submodule update --init --checkout --recursive'
  sh 'git submodule foreach --recursive git reset --hard'
  sh 'git submodule foreach --recursive git clean -d -x -f'
end

desc 'Hard clean repository'
task :git_clean do
  sh 'git reset --hard'
  sh 'git clean -d -x -f'
end

desc 'Install optional ThirdParties, if present'
task :install_3rd do
  if Dir.exist?('ThirdParties')
    Dir.chdir('ThirdParties') { sh 'rake install' }
  else
    puts 'ThirdParties directory not found; skipping'.yellow
  end
end

desc 'Generate compile_commands.json and run cppcheck'
task :cppcheck do
  FileUtils.rm_rf('build')
  in_dir('build') do
    sh 'cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..'
    sh 'cppcheck --project=compile_commands.json'
  end
end

desc 'Run CPack from build/'
task :cpack do
  Dir.chdir('build') do
    sh 'cpack -C CPackConfig.cmake'
    sh 'cpack -C CPackSourceConfig.cmake'
  end
end

# Compatibility aliases.
task build_osx: :build
task build_linux: :build
task build_mingw: :build
task build_win: :build
task clean_osx: :clean
task clean_linux: :clean
task clean_mingw: :clean
task clean_win: :clean
